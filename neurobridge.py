import json
import requests
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CORE PIPELINE ENGINE
# ============================================================================


class PipelineContext:
    """Stores data flowing through the pipeline"""

    def __init__(self):
        self.data = {}
        self.history = []

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.history.append({"key": key, "value": value})

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def update(self, data: Dict):
        self.data.update(data)


class ComponentBase:
    """Base class for all pipeline components"""

    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "unnamed")

    def execute(self, context: PipelineContext) -> Any:
        raise NotImplementedError


# ============================================================================
# LLM COMPONENTS (OpenAPI Compatible)
# ============================================================================


class OpenAPILLM(ComponentBase):
    """Universal OpenAPI-compatible LLM component"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_base = config["api_base"]
        self.api_key = config.get("api_key", "")
        self.model = config["model"]
        self.system_prompt = config.get("system_prompt", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.available_functions = config.get("available_functions", [])

    def execute(self, context: PipelineContext) -> str:
        # Build messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Get user input from context
        user_input = context.get("input", "")
        if isinstance(user_input, dict):
            # Handle multimodal input (text + images)
            content = []
            if "text" in user_input:
                content.append({"type": "text", "text": user_input["text"]})
            if "images" in user_input:
                for img_data in user_input["images"]:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        }
                    )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": str(user_input)})

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add function calling if available
        if self.available_functions:
            payload["tools"] = [
                {"type": "function", "function": f} for f in self.available_functions
            ]

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        message = result["choices"][0]["message"]

        # Handle function calls
        if message.get("tool_calls"):
            context.set("function_calls", message["tool_calls"])
            return message.get("content", "")

        return message["content"]


# ============================================================================
# VISION COMPONENTS
# ============================================================================


class VisionModel(ComponentBase):
    """Vision model for image captioning/analysis"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_base = config["api_base"]
        self.api_key = config.get("api_key", "")
        self.model = config["model"]

    def execute(self, context: PipelineContext) -> str:
        image_data = context.get("image")
        if not image_data:
            raise ValueError("No image data in context")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.config.get("prompt", "Describe this image"),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.config.get("max_tokens", 300),
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]


# ============================================================================
# AUDIO COMPONENTS
# ============================================================================


class WhisperSTT(ComponentBase):
    """Whisper Speech-to-Text (local)"""

    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            import whisper

            self.model = whisper.load_model(config.get("model_size", "base"))
        except ImportError:
            raise ImportError("Install openai-whisper: pip install openai-whisper")

    def execute(self, context: PipelineContext) -> str:
        audio_path = context.get("audio_file")
        if not audio_path:
            raise ValueError("No audio file in context")

        result = self.model.transcribe(audio_path)
        return result["text"]


class KokoroTTS(ComponentBase):
    """Kokoro Text-to-Speech (local)"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.output_path = config.get("output_path", "output.wav")
        # Initialize Kokoro model here
        # Note: Kokoro setup depends on specific implementation

    def execute(self, context: PipelineContext) -> str:
        text = context.get("text", context.get("input", ""))

        # Kokoro TTS generation code
        # This is a placeholder - actual implementation depends on Kokoro API
        print(f"[Kokoro TTS] Generating speech for: {text[:50]}...")

        # Generate audio and save to output_path
        # kokoro_generate(text, self.output_path)

        return self.output_path


# ============================================================================
# CUSTOM ML MODELS
# ============================================================================


class CustomPickleModel(ComponentBase):
    """Load and run custom pickle models"""

    def __init__(self, config: Dict):
        super().__init__(config)
        with open(config["model_path"], "rb") as f:
            self.model = pickle.load(f)
        self.input_keys = config.get("input_keys", [])
        self.output_key = config.get("output_key", "prediction")

    def execute(self, context: PipelineContext) -> Any:
        # Gather inputs
        inputs = []
        for key in self.input_keys:
            val = context.get(key)
            if val is None:
                raise ValueError(f"Missing input: {key}")
            inputs.append(val)

        # Convert to numpy array if needed
        X = np.array([inputs])

        # Run prediction
        prediction = self.model.predict(X)[0]
        context.set(self.output_key, prediction)

        return prediction


class CustomTensorFlowModel(ComponentBase):
    """Load and run TensorFlow models"""

    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(config["model_path"])
        except ImportError:
            raise ImportError("Install tensorflow: pip install tensorflow")
        self.input_keys = config.get("input_keys", [])
        self.output_key = config.get("output_key", "prediction")

    def execute(self, context: PipelineContext) -> Any:
        inputs = []
        for key in self.input_keys:
            val = context.get(key)
            if val is None:
                raise ValueError(f"Missing input: {key}")
            inputs.append(val)

        X = np.array([inputs])
        prediction = self.model.predict(X, verbose=0)[0]

        # Handle different output formats
        if len(prediction.shape) > 0 and prediction.shape[0] > 1:
            result = np.argmax(prediction)
        else:
            result = float(prediction)

        context.set(self.output_key, result)
        return result


# ============================================================================
# HARDWARE COMPONENTS
# ============================================================================


class HardwareSensor(ComponentBase):
    """Interface with hardware sensors via REST API"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.esp32_url = config["esp32_url"]
        self.sensor_type = config["sensor_type"]
        self.endpoint = config.get("endpoint", "/sensor/read")

    def execute(self, context: PipelineContext) -> Dict:
        response = requests.get(f"{self.esp32_url}{self.endpoint}", timeout=5)
        response.raise_for_status()
        data = response.json()

        # Store sensor data in context
        for key, value in data.items():
            context.set(key, value)

        return data


class HardwareActuator(ComponentBase):
    """Control hardware actuators via REST API"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.esp32_url = config["esp32_url"]
        self.actuator_type = config["actuator_type"]
        self.endpoint = config.get("endpoint", "/actuator/control")

    def execute(self, context: PipelineContext) -> Dict:
        # Get control parameters from context
        params = {}
        for key in self.config.get("param_keys", []):
            val = context.get(key)
            if val is not None:
                params[key] = val

        response = requests.post(
            f"{self.esp32_url}{self.endpoint}", json=params, timeout=5
        )
        response.raise_for_status()
        return response.json()


class CustomHardwareEndpoint(ComponentBase):
    """Call custom hardware endpoints"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.esp32_url = config["esp32_url"]
        self.endpoint = config["endpoint"]
        self.method = config.get("method", "POST")

    def execute(self, context: PipelineContext) -> Dict:
        params = {}
        for key in self.config.get("param_keys", []):
            val = context.get(key)
            if val is not None:
                params[key] = val

        if self.method == "POST":
            response = requests.post(
                f"{self.esp32_url}{self.endpoint}", json=params, timeout=5
            )
        else:
            response = requests.get(
                f"{self.esp32_url}{self.endpoint}", params=params, timeout=5
            )

        response.raise_for_status()
        return response.json()


# ============================================================================
# COMPONENT FACTORY
# ============================================================================


class ComponentFactory:
    """Creates components based on configuration"""

    @staticmethod
    def create(config: Dict) -> ComponentBase:
        component_type = config["type"]

        if component_type == "openapi_llm":
            return OpenAPILLM(config)
        elif component_type == "vision_model":
            return VisionModel(config)
        elif component_type == "whisper_stt":
            return WhisperSTT(config)
        elif component_type == "kokoro_tts":
            return KokoroTTS(config)
        elif component_type == "pickle_model":
            return CustomPickleModel(config)
        elif component_type == "tensorflow_model":
            return CustomTensorFlowModel(config)
        elif component_type == "hardware_sensor":
            return HardwareSensor(config)
        elif component_type == "hardware_actuator":
            return HardwareActuator(config)
        elif component_type == "custom_endpoint":
            return CustomHardwareEndpoint(config)
        else:
            raise ValueError(f"Unknown component type: {component_type}")


# ============================================================================
# PIPELINE EXECUTOR
# ============================================================================


class Pipeline:
    """Executes a sequence of components"""

    def __init__(self, config: Dict):
        self.config = config
        self.components = []

        for comp_config in config["pipeline"]:
            component = ComponentFactory.create(comp_config)
            self.components.append(component)

    def execute(self, initial_data: Dict = None) -> PipelineContext:
        context = PipelineContext()

        if initial_data:
            context.update(initial_data)

        for component in self.components:
            try:
                print(f"[Pipeline] Executing: {component.name}")
                result = component.execute(context)

                # Store result in context with component name
                context.set(f"{component.name}_output", result)

                # Update input for next component
                context.set("input", result)

            except Exception as e:
                print(f"[Pipeline] Error in {component.name}: {str(e)}")
                raise

        return context


# ============================================================================
# MAIN FRAMEWORK
# ============================================================================


class AIHardwareFramework:
    """Main framework class"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.pipelines = {}
        for name, pipeline_config in self.config.get("pipelines", {}).items():
            self.pipelines[name] = Pipeline(pipeline_config)

    def run_pipeline(
        self, pipeline_name: str, initial_data: Dict = None
    ) -> PipelineContext:
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_name}")

        print(f"\n{'=' * 60}")
        print(f"Running Pipeline: {pipeline_name}")
        print(f"{'=' * 60}\n")

        return self.pipelines[pipeline_name].execute(initial_data)

    def generate_esp32_firmware(self, output_path: str = "esp32_firmware.ino"):
        """Generate ESP32 firmware with REST API endpoints"""
        from esp32_firmware_generator import generate_firmware

        generate_firmware(self.config, output_path)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load framework
    framework = AIHardwareFramework("config.json")

    # Example 1: Image + Voice to LLM to TTS
    print("\n=== Example 1: Multimodal Pipeline ===")
    result = framework.run_pipeline(
        "multimodal_pipeline",
        {"audio_file": "input.wav", "image": "base64_encoded_image_data"},
    )

    # Example 2: DHT22 -> ML Model -> LLM -> Hardware
    print("\n=== Example 2: Sensor-ML-LLM Pipeline ===")
    result = framework.run_pipeline("climate_control_pipeline")

    print("\n[Framework] Pipeline execution complete!")
    print(f"[Framework] Final context: {result.data}")
