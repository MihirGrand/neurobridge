from typing import Dict, List
from pathlib import Path

def generate_firmware(config: Dict, output_path: str):
    """Generate complete ESP32 firmware based on configuration"""

    hardware_config = config.get('hardware', {})
    sensors = hardware_config.get('sensors', [])
    actuators = hardware_config.get('actuators', [])
    custom_endpoints = hardware_config.get('custom_endpoints', [])

    firmware = []

    # Header
    firmware.append("""/*
 * Auto-generated ESP32 Firmware
 * Modular AI/Hardware Pipeline Framework
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
""")

    # Add hardware-specific libraries
    libs = set()
    for sensor in sensors:
        libs.update(get_sensor_libraries(sensor['type']))
    for actuator in actuators:
        libs.update(get_actuator_libraries(actuator['type']))

    for lib in libs:
        firmware.append(f"#include <{lib}>\n")

    firmware.append("\n")

    # WiFi configuration
    wifi_config = hardware_config.get('wifi', {})
    firmware.append(f"""
// WiFi Configuration
const char* ssid = "{wifi_config.get('ssid', 'YOUR_SSID')}";
const char* password = "{wifi_config.get('password', 'YOUR_PASSWORD')}";

WebServer server({wifi_config.get('port', 80)});
""")

    # Hardware pin definitions
    firmware.append("\n// Hardware Pin Definitions\n")
    for sensor in sensors:
        firmware.append(generate_sensor_pins(sensor))
    for actuator in actuators:
        firmware.append(generate_actuator_pins(actuator))

    # Hardware objects
    firmware.append("\n// Hardware Objects\n")
    for sensor in sensors:
        firmware.append(generate_sensor_objects(sensor))
    for actuator in actuators:
        firmware.append(generate_actuator_objects(actuator))

    # Periodic monitoring variables
    for sensor in sensors:
        if sensor.get('periodic_monitoring', {}).get('enabled'):
            firmware.append(f"""
unsigned long last{sensor['name']}Check = 0;
const unsigned long {sensor['name']}Interval = {sensor['periodic_monitoring'].get('interval_ms', 5000)};
float last{sensor['name']}Temp = 0;
float last{sensor['name']}Humidity = 0;
""")

    # Setup function
    firmware.append("""
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize hardware
""")

    for sensor in sensors:
        firmware.append(generate_sensor_setup(sensor))
    for actuator in actuators:
        firmware.append(generate_actuator_setup(actuator))

    firmware.append("""
  // Connect to WiFi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Setup REST API endpoints
""")

    # Register endpoints
    for sensor in sensors:
        firmware.append(f"  server.on(\"/sensor/{sensor['name']}/read\", HTTP_GET, handle{sensor['name']}Read);\n")

    for actuator in actuators:
        firmware.append(f"  server.on(\"/actuator/{actuator['name']}/control\", HTTP_POST, handle{actuator['name']}Control);\n")

    for endpoint in custom_endpoints:
        method = endpoint.get('method', 'POST')
        firmware.append(f"  server.on(\"{endpoint['path']}\", HTTP_{method}, handle{endpoint['name']});\n")

    firmware.append("""
  server.begin();
  Serial.println("HTTP server started");
}
""")

    # Loop function
    firmware.append("""
void loop() {
  server.handleClient();

""")

    # Add periodic monitoring
    for sensor in sensors:
        if sensor.get('periodic_monitoring', {}).get('enabled'):
            mode = sensor['periodic_monitoring'].get('mode', 'interval')
            trigger_pipeline = sensor['periodic_monitoring'].get('trigger_pipeline', '')

            if mode == 'interval':
                firmware.append(f"""
  // Periodic monitoring for {sensor['name']}
  if (millis() - last{sensor['name']}Check >= {sensor['name']}Interval) {{
    last{sensor['name']}Check = millis();
    check{sensor['name']}AndTrigger();
  }}
""")
            elif mode == 'on_change':
                firmware.append(f"""
  // On-change monitoring for {sensor['name']}
  check{sensor['name']}Change();
""")

    firmware.append("}\n\n")

    # Handler functions for sensors
    for sensor in sensors:
        firmware.append(generate_sensor_handlers(sensor))

    # Handler functions for actuators
    for actuator in actuators:
        firmware.append(generate_actuator_handlers(actuator))

    # Custom endpoint handlers
    for endpoint in custom_endpoints:
        firmware.append(generate_custom_handler(endpoint))

    # Monitoring functions
    for sensor in sensors:
        if sensor.get('periodic_monitoring', {}).get('enabled'):
            firmware.append(generate_monitoring_functions(sensor))

    # Write to file
    with open(output_path, 'w') as f:
        f.write(''.join(firmware))

    print(f"[Firmware Generator] Generated: {output_path}")

    # Generate separate hardware interface files
    generate_hardware_interfaces(sensors, actuators, custom_endpoints)


def get_sensor_libraries(sensor_type: str) -> List[str]:
    """Get required libraries for sensor type"""
    libs = {
        'DHT22': ['DHT.h'],
        'INMP441': ['I2S.h'],
        'ESP32-CAM': ['esp_camera.h']
    }
    return libs.get(sensor_type, [])


def get_actuator_libraries(actuator_type: str) -> List[str]:
    """Get required libraries for actuator type"""
    libs = {
        'SH1106': ['Wire.h', 'Adafruit_GFX.h', 'Adafruit_SH110X.h'],
        'SG90': ['ESP32Servo.h']
    }
    return libs.get(actuator_type, [])


def generate_sensor_pins(sensor: Dict) -> str:
    """Generate pin definitions for sensor"""
    sensor_type = sensor['type']
    name = sensor['name']
    pins = sensor.get('pins', {})

    if sensor_type == 'DHT22':
        return f"#define {name.upper()}_PIN {pins.get('data', 4)}\n"
    elif sensor_type == 'INMP441':
        return f"""#define {name.upper()}_SCK {pins.get('sck', 14)}
#define {name.upper()}_WS {pins.get('ws', 15)}
#define {name.upper()}_SD {pins.get('sd', 32)}
"""
    return ""


def generate_actuator_pins(actuator: Dict) -> str:
    """Generate pin definitions for actuator"""
    actuator_type = actuator['type']
    name = actuator['name']
    pins = actuator.get('pins', {})

    if actuator_type == 'SH1106':
        return f"""#define {name.upper()}_SDA {pins.get('sda', 21)}
#define {name.upper()}_SCL {pins.get('scl', 22)}
"""
    elif actuator_type == 'SG90':
        return f"#define {name.upper()}_PIN {pins.get('control', 13)}\n"
    return ""


def generate_sensor_objects(sensor: Dict) -> str:
    """Generate object instantiation for sensor"""
    sensor_type = sensor['type']
    name = sensor['name']

    if sensor_type == 'DHT22':
        return f"DHT {name}({name.upper()}_PIN, DHT22);\n"
    elif sensor_type == 'INMP441':
        return f"// I2S microphone {name} initialized in setup\n"
    return ""


def generate_actuator_objects(actuator: Dict) -> str:
    """Generate object instantiation for actuator"""
    actuator_type = actuator['type']
    name = actuator['name']

    if actuator_type == 'SH1106':
        return f"Adafruit_SH1106G {name} = Adafruit_SH1106G(128, 64, &Wire, -1);\n"
    elif actuator_type == 'SG90':
        return f"Servo {name};\n"
    return ""


def generate_sensor_setup(sensor: Dict) -> str:
    """Generate setup code for sensor"""
    sensor_type = sensor['type']
    name = sensor['name']

    if sensor_type == 'DHT22':
        return f"""  {name}.begin();
  Serial.println("{name} initialized");
"""
    elif sensor_type == 'INMP441':
        return f"""  // Initialize I2S for {name}
  I2S.setAllPins({name.upper()}_SCK, {name.upper()}_WS, {name.upper()}_SD, -1, -1);
  I2S.begin(I2S_PHILIPS_MODE, 16000, 16);
  Serial.println("{name} initialized");
"""
    return ""


def generate_actuator_setup(actuator: Dict) -> str:
    """Generate setup code for actuator"""
    actuator_type = actuator['type']
    name = actuator['name']

    if actuator_type == 'SH1106':
        return f"""  {name}.begin(0x3C, true);
  {name}.clearDisplay();
  {name}.setTextSize(1);
  {name}.setTextColor(SH110X_WHITE);
  {name}.setCursor(0, 0);
  {name}.println("System Ready");
  {name}.display();
  Serial.println("{name} initialized");
"""
    elif actuator_type == 'SG90':
        return f"""  {name}.attach({name.upper()}_PIN);
  {name}.write(0);
  Serial.println("{name} initialized");
"""
    return ""


def generate_sensor_handlers(sensor: Dict) -> str:
    """Generate REST API handlers for sensor"""
    sensor_type = sensor['type']
    name = sensor['name']

    if sensor_type == 'DHT22':
        return f"""
void handle{name}Read() {{
  float temperature = {name}.readTemperature();
  float humidity = {name}.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {{
    server.send(500, "application/json", "{{\\\"error\\\":\\\"Failed to read sensor\\\"}}");
    return;
  }}

  StaticJsonDocument<200> doc;
  doc["temperature"] = temperature;
  doc["humidity"] = humidity;
  doc["sensor"] = "{name}";
  doc["type"] = "DHT22";

  String response;
  serializeJson(doc, response);
  server.send(200, "application/json", response);
}}
"""
    return ""


def generate_actuator_handlers(actuator: Dict) -> str:
    """Generate REST API handlers for actuator"""
    actuator_type = actuator['type']
    name = actuator['name']

    if actuator_type == 'SH1106':
        return f"""
void handle{name}Control() {{
  if (!server.hasArg("plain")) {{
    server.send(400, "application/json", "{{\\\"error\\\":\\\"No data\\\"}}");
    return;
  }}

  StaticJsonDocument<500> doc;
  deserializeJson(doc, server.arg("plain"));

  {name}.clearDisplay();
  {name}.setCursor(0, 0);

  if (doc.containsKey("text")) {{
    const char* text = doc["text"];
    {name}.println(text);
  }}

  {name}.display();

  server.send(200, "application/json", "{{\\\"status\\\":\\\"ok\\\"}}");
}}
"""
    elif actuator_type == 'SG90':
        return f"""
void handle{name}Control() {{
  if (!server.hasArg("plain")) {{
    server.send(400, "application/json", "{{\\\"error\\\":\\\"No data\\\"}}");
    return;
  }}

  StaticJsonDocument<200> doc;
  deserializeJson(doc, server.arg("plain"));

  if (doc.containsKey("angle")) {{
    int angle = doc["angle"];
    angle = constrain(angle, 0, 180);
    {name}.write(angle);

    StaticJsonDocument<200> response;
    response["status"] = "ok";
    response["angle"] = angle;

    String responseStr;
    serializeJson(response, responseStr);
    server.send(200, "application/json", responseStr);
  }} else {{
    server.send(400, "application/json", "{{\\\"error\\\":\\\"Missing angle\\\"}}");
  }}
}}
"""
    return ""


def generate_custom_handler(endpoint: Dict) -> str:
    """Generate custom endpoint handler from template"""
    name = endpoint['name']
    template_path = endpoint.get('template_file', f"hardware_interfaces/{name}.cpp")

    # Load template if exists
    if Path(template_path).exists():
        with open(template_path, 'r') as f:
            return f.read() + "\n"

    # Generate basic handler
    return f"""
void handle{name}() {{
  // Custom endpoint: {name}
  // TODO: Implement custom logic
  // Template file: {template_path}

  server.send(200, "application/json", "{{\\\"status\\\":\\\"ok\\\"}}");
}}
"""


def generate_monitoring_functions(sensor: Dict) -> str:
    """Generate periodic monitoring functions"""
    name = sensor['name']
    mode = sensor.get('periodic_monitoring', {}).get('mode', 'interval')
    trigger_url = sensor.get('periodic_monitoring', {}).get('trigger_pipeline_url', '')

    if mode == 'interval':
        return f"""
void check{name}AndTrigger() {{
  float temperature = {name}.readTemperature();
  float humidity = {name}.readHumidity();

  if (!isnan(temperature) && !isnan(humidity)) {{
    Serial.printf("{name} - Temp: %.1f°C, Humidity: %.1f%%\\n", temperature, humidity);

    // TODO: Send HTTP request to trigger pipeline at {trigger_url}
    // Include sensor data in request body
  }}
}}
"""
    elif mode == 'on_change':
        threshold = sensor.get('periodic_monitoring', {}).get('change_threshold', 1.0)
        return f"""
void check{name}Change() {{
  float temperature = {name}.readTemperature();
  float humidity = {name}.readHumidity();

  if (!isnan(temperature) && !isnan(humidity)) {{
    if (abs(temperature - last{name}Temp) >= {threshold} ||
        abs(humidity - last{name}Humidity) >= {threshold}) {{

      Serial.printf("{name} - Change detected! Temp: %.1f°C, Humidity: %.1f%%\\n",
                    temperature, humidity);

      last{name}Temp = temperature;
      last{name}Humidity = humidity;

      // TODO: Send HTTP request to trigger pipeline at {trigger_url}
    }}
  }}
}}
"""
    return ""


def generate_hardware_interfaces(sensors: List[Dict], actuators: List[Dict], custom_endpoints: List[Dict]):
    """Generate separate interface files for user customization"""

    Path("hardware_interfaces").mkdir(exist_ok=True)

    # Generate sensor interfaces
    for sensor in sensors:
        filename = f"hardware_interfaces/{sensor['name']}_interface.cpp"
        if not Path(filename).exists():
            with open(filename, 'w') as f:
                f.write(f"""/*
 * {sensor['name']} Interface
 * Type: {sensor['type']}
 *
 * Edit this file to customize {sensor['name']} behavior
 */

// Add custom functions for {sensor['name']} here
""")

    # Generate actuator interfaces
    for actuator in actuators:
        filename = f"hardware_interfaces/{actuator['name']}_interface.cpp"
        if not Path(filename).exists():
            with open(filename, 'w') as f:
                f.write(f"""/*
 * {actuator['name']} Interface
 * Type: {actuator['type']}
 *
 * Edit this file to customize {actuator['name']} behavior
 */

// Add custom functions for {actuator['name']} here
""")

    # Generate custom endpoint templates
    for endpoint in custom_endpoints:
        filename = f"hardware_interfaces/{endpoint['name']}.cpp"
        if not Path(filename).exists():
            with open(filename, 'w') as f:
                f.write(generate_custom_endpoint_template(endpoint))

    print("[Firmware Generator] Hardware interface files created in hardware_interfaces/")


def generate_custom_endpoint_template(endpoint: Dict) -> str:
    """Generate template for custom endpoint"""
    name = endpoint['name']
    description = endpoint.get('description', f'Custom endpoint {name}')

    if name == 'showSmileExpression':
        return """/*
 * Custom Endpoint: Show Smile Expression
 * Displays a smiley face on OLED display
 */

void handleshowSmileExpression() {
  display.clearDisplay();

  // Draw smiley face
  display.fillCircle(64, 32, 25, SH110X_WHITE);  // Face
  display.fillCircle(54, 25, 3, SH110X_BLACK);   // Left eye
  display.fillCircle(74, 25, 3, SH110X_BLACK);   // Right eye

  // Smile
  for (int i = 0; i < 20; i++) {
    int x = 50 + i;
    int y = 35 + (i - 10) * (i - 10) / 20;
    display.drawPixel(x, y, SH110X_BLACK);
  }

  display.display();

  server.send(200, "application/json", "{\\"status\\":\\"ok\\",\\"expression\\":\\"smile\\"}");
}
"""

    return f"""/*
 * Custom Endpoint: {name}
 * {description}
 */

void handle{name}() {{
  // Parse request
  if (server.hasArg("plain")) {{
    StaticJsonDocument<500> doc;
    deserializeJson(doc, server.arg("plain"));

    // TODO: Implement custom logic here

  }}

  // Send response
  server.send(200, "application/json", "{{\\\"status\\\":\\\"ok\\\"}}");
}}
"""
