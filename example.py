from neurobridge import AIHardwareFramework
import json
from pathlib import Path


def setup_example_1():
    """Example 1: Multimodal Pipeline (Image + Voice → LLM → TTS)"""

    print("\n" + "=" * 60)
    print("Example 1: Multimodal Pipeline Setup")
    print("=" * 60)

    config = {
        "description": "Image + Voice → LLM → TTS Pipeline",
        "pipelines": {
            "multimodal_pipeline": {
                "pipeline": [
                    {
                        "type": "whisper_stt",
                        "name": "voice_transcription",
                        "model_size": "base",
                    },
                    {
                        "type": "vision_model",
                        "name": "image_caption",
                        "api_base": "https://api.openai.com/v1",
                        "api_key": "YOUR_OPENAI_KEY",
                        "model": "gpt-4-vision-preview",
                        "prompt": "Describe what you see in this image in detail",
                    },
                    {
                        "type": "openapi_llm",
                        "name": "response_generator",
                        "api_base": "https://api.anthropic.com/v1",
                        "api_key": "YOUR_ANTHROPIC_KEY",
                        "model": "claude-sonnet-4-5-20250929",
                        "system_prompt": "You are a helpful assistant. The user said something (transcribed) and showed you an image (described). Respond naturally and conversationally in 2-3 sentences.",
                        "temperature": 0.7,
                        "max_tokens": 500,
                    },
                    {
                        "type": "kokoro_tts",
                        "name": "speech_synthesis",
                        "model_path": "models/kokoro-v0.19.pth",
                        "output_path": "output/response.wav",
                    },
                ]
            }
        },
        "hardware": {
            "wifi": {
                "ssid": "YOUR_WIFI_SSID",
                "password": "YOUR_WIFI_PASSWORD",
                "port": 80,
            },
            "sensors": [],
            "actuators": [],
            "custom_endpoints": [],
        },
    }

    # Save config
    with open("config_multimodal.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✓ Configuration saved to config_multimodal.json")
    print("\nTo use this pipeline:")
    print("1. Install Whisper: pip install openai-whisper")
    print("2. Set up Kokoro TTS model")
    print("3. Add your API keys to the config file")
    print("4. Run:")
    print("   framework = AIHardwareFramework('config_multimodal.json')")
    print("   result = framework.run_pipeline('multimodal_pipeline', {")
    print("       'audio_file': 'path/to/audio.wav',")
    print("       'image': 'base64_encoded_image_data'")
    print("   })")


def setup_example_2():
    """Example 2: Climate Control Pipeline (DHT22 → ML → LLM → Hardware)"""

    print("\n" + "=" * 60)
    print("Example 2: Climate Control Pipeline Setup")
    print("=" * 60)

    config = {
        "description": "DHT22 → ML Model → LLM → OLED + Servo Pipeline",
        "pipelines": {
            "climate_control_pipeline": {
                "pipeline": [
                    {
                        "type": "hardware_sensor",
                        "name": "climate_sensor",
                        "esp32_url": "http://192.168.1.100",
                        "sensor_type": "DHT22",
                        "endpoint": "/sensor/dht22/read",
                    },
                    {
                        "type": "pickle_model",
                        "name": "climate_classifier",
                        "model_path": "models/climate_classifier.pkl",
                        "input_keys": ["temperature", "humidity"],
                        "output_key": "climate_class",
                    },
                    {
                        "type": "openapi_llm",
                        "name": "summary_generator",
                        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
                        "api_key": "YOUR_GEMINI_KEY",
                        "model": "gemini-2.0-flash-exp",
                        "system_prompt": "Given climate classification (0=Not Hot, 1=Hot, 2=Hot and Humid) and temperature/humidity data, provide exactly 1 short sentence describing the conditions.",
                        "temperature": 0.5,
                        "max_tokens": 50,
                    },
                    {
                        "type": "custom_endpoint",
                        "name": "display_climate",
                        "esp32_url": "http://192.168.1.100",
                        "endpoint": "/custom/displayClimate",
                        "method": "POST",
                        "param_keys": [
                            "climate_class",
                            "summary_generator_output",
                            "temperature",
                            "humidity",
                        ],
                    },
                ]
            }
        },
        "hardware": {
            "wifi": {
                "ssid": "YOUR_WIFI_SSID",
                "password": "YOUR_WIFI_PASSWORD",
                "port": 80,
            },
            "sensors": [
                {
                    "name": "dht22",
                    "type": "DHT22",
                    "pins": {"data": 4},
                    "periodic_monitoring": {
                        "enabled": True,
                        "mode": "interval",
                        "interval_ms": 10000,
                        "trigger_pipeline": "climate_control_pipeline",
                        "trigger_pipeline_url": "http://YOUR_SERVER:5000/api/trigger/climate_control_pipeline",
                    },
                }
            ],
            "actuators": [
                {"name": "display", "type": "SH1106", "pins": {"sda": 21, "scl": 22}},
                {"name": "servo", "type": "SG90", "pins": {"control": 13}},
            ],
            "custom_endpoints": [
                {
                    "name": "displayClimate",
                    "path": "/custom/displayClimate",
                    "method": "POST",
                    "description": "Display climate info on OLED with appropriate icon and control servo",
                    "template_file": "hardware_interfaces/displayClimate.cpp",
                },
                {
                    "name": "showSmileExpression",
                    "path": "/custom/smile",
                    "method": "GET",
                    "description": "Show smiley face on display",
                    "template_file": "hardware_interfaces/showSmileExpression.cpp",
                },
            ],
        },
    }

    # Save config
    with open("config_climate.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✓ Configuration saved to config_climate.json")

    # Train the model
    print("\n📊 Training climate classifier model...")
    try:
        import subprocess

        subprocess.run(["python", "train_climate_model.py"], check=True)
        print("✓ Model trained successfully")
    except:
        print("⚠ Run 'python train_climate_model.py' to train the model")

    print("\nTo use this pipeline:")
    print("1. Train the model: python train_climate_model.py")
    print("2. Generate ESP32 firmware:")
    print("   framework = AIHardwareFramework('config_climate.json')")
    print("   framework.generate_esp32_firmware('esp32_climate.ino')")
    print("3. Upload esp32_climate.ino to your ESP32")
    print("4. Update ESP32 IP address in config")
    print("5. Add your Gemini API key to config")
    print("6. Run the pipeline:")
    print("   result = framework.run_pipeline('climate_control_pipeline')")


def demonstrate_pipeline_flow():
    """Show how data flows through the climate pipeline"""

    print("\n" + "=" * 60)
    print("Climate Control Pipeline Flow")
    print("=" * 60)

    print("""
Step 1: DHT22 Sensor Reading
   ↓
   GET http://192.168.1.100/sensor/dht22/read
   Response: {"temperature": 30.5, "humidity": 85.0}

Step 2: ML Model Classification
   ↓
   Input: [30.5, 85.0]
   Model: climate_classifier.pkl
   Output: 2 (Hot and Humid)

Step 3: LLM Summary Generation
   ↓
   To Gemini: "Given climate classification (2=Hot and Humid)
              and temperature 30.5°C, humidity 85%..."
   Response: "Uncomfortably hot and humid conditions."

Step 4: Display on OLED with Icon
   ↓
   POST http://192.168.1.100/custom/displayClimate
   Body: {
     "climate_class": 2,
     "summary_generator_output": "Uncomfortably hot...",
     "temperature": 30.5,
     "humidity": 85.0
   }

   OLED shows:
   ┌────────────────┐
   │ ☀💧 Hot &      │
   │    Humid       │
   │                │
   │ Uncomfortably  │
   │ hot and humid  │
   │ conditions.    │
   └────────────────┘

   Servo: Moves to 90° (high speed fan)

Classification Rules:
- Class 0 (Not Hot):     temp < 25°C              → Servo at 0°
- Class 1 (Hot):         25°C ≤ temp, humid < 70% → Servo at 45°
- Class 2 (Hot & Humid): 25°C ≤ temp, humid ≥ 70% → Servo at 90°
""")


def create_project_structure():
    """Create necessary directories"""

    print("\n" + "=" * 60)
    print("Creating Project Structure")
    print("=" * 60)

    dirs = ["models", "output", "hardware_interfaces"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"✓ Created directory: {d}/")

    print("\nProject structure:")
    print("""
project/
├── main_framework.py          # Core framework
├── esp32_firmware_generator.py # Firmware generator
├── train_climate_model.py     # Model training
├── usage_example.py           # This file
├── config_multimodal.json     # Example 1 config
├── config_climate.json        # Example 2 config
├── models/
│   └── climate_classifier.pkl # Trained ML model
├── output/
│   └── response.wav           # Generated audio
└── hardware_interfaces/
    ├── displayClimate.cpp     # Custom endpoint
    └── showSmileExpression.cpp # Custom endpoint
""")


def main():
    """Run all setup examples"""

    print("""
╔════════════════════════════════════════════════════════════╗
║   Modular AI/Hardware Pipeline Framework - Setup Guide    ║
╚════════════════════════════════════════════════════════════╝
""")

    create_project_structure()
    # setup_example_1()
    setup_example_2()
    demonstrate_pipeline_flow()

    print("\n" + "=" * 60)
    print("Quick Start Commands")
    print("=" * 60)
    print("""
# Install dependencies
pip install requests pillow numpy scikit-learn

# Optional: Install Whisper for Example 1
pip install openai-whisper

# Train climate model for Example 2
python train_climate_model.py

# Generate ESP32 firmware for Example 2
python -c "
from main_framework import AIHardwareFramework
framework = AIHardwareFramework('config_climate.json')
framework.generate_esp32_firmware('esp32_climate.ino')
"

# Run Example 2 (after ESP32 setup)
python -c "
from main_framework import AIHardwareFramework
framework = AIHardwareFramework('config_climate.json')
result = framework.run_pipeline('climate_control_pipeline')
print(result.data)
"
""")

    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. Review config_multimodal.json and config_climate.json
2. Add your API keys (OpenAI, Anthropic, Gemini)
3. For hardware setup:
   - Upload generated .ino file to ESP32
   - Connect DHT22 to pin 4
   - Connect SH1106 OLED to I2C (SDA=21, SCL=22)
   - Connect SG90 Servo to pin 13
4. Update ESP32 IP address in config
5. Run your pipelines!

For detailed documentation, see the README.
""")


if __name__ == "__main__":
    main()
