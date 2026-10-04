#include <Arduino.h>
#include "Gimbal.h"

namespace {
constexpr int SERVO_YAW_PIN = 13;
constexpr int SERVO_PITCH_PIN = 15;
constexpr int LASER_PIN = 18;
constexpr unsigned long SERIAL_BAUD_RATE = 115200;

constexpr uint8_t FRAME_HEADER = 0x15;
constexpr uint8_t DEVICE_YAW = 0x01;
constexpr uint8_t DEVICE_PITCH = 0x02;
constexpr uint8_t DEVICE_LASER = 0x03;
constexpr uint8_t DEVICE_STOP = 0x04;
// Fine moves carry two value bytes: hundredths of a degree, high byte first.
constexpr uint8_t DEVICE_YAW_FINE = 0x05;
constexpr uint8_t DEVICE_PITCH_FINE = 0x06;
// Travel speed in degrees per second (1-255).
constexpr uint8_t DEVICE_SPEED = 0x07;
// Replies with FIRMWARE_ID so the host can detect fine-move support.
constexpr uint8_t DEVICE_PING = 0x08;

constexpr char FIRMWARE_ID[] = "FIXPOINT 1";
constexpr uint16_t MAXIMUM_CENTIDEGREES = 18000;

// Adjust the angle limits to match the safe mechanical travel of your gimbal.
constexpr GimbalServoConfig YAW_CONFIG = {
    SERVO_YAW_PIN,
    0.0f,
    180.0f,
    90.0f,
    500,
    2500,
};

constexpr GimbalServoConfig PITCH_CONFIG = {
    SERVO_PITCH_PIN,
    0.0f,
    180.0f,
    90.0f,
    500,
    2500,
};

Gimbal gimbal(YAW_CONFIG, PITCH_CONFIG, 60.0f);

enum class ParserState : uint8_t {
  WaitingForHeader,
  WaitingForDevice,
  WaitingForValue,
  WaitingForLowByte,
};

ParserState parserState = ParserState::WaitingForHeader;
uint8_t pendingDevice = 0;
uint8_t pendingHighByte = 0;

bool isDevice(uint8_t value) {
  return value >= DEVICE_YAW && value <= DEVICE_PING;
}

bool isFineDevice(uint8_t device) {
  return device == DEVICE_YAW_FINE || device == DEVICE_PITCH_FINE;
}

void applyFineCommand(uint8_t device, uint16_t centidegrees) {
  if (centidegrees > MAXIMUM_CENTIDEGREES) {
    return;
  }
  const GimbalAxis axis =
      device == DEVICE_YAW_FINE ? GimbalAxis::Yaw : GimbalAxis::Pitch;
  gimbal.moveTo(axis, centidegrees / 100.0f);
}

void applyCommand(uint8_t device, uint8_t value) {
  if (device == DEVICE_YAW && value <= 180) {
    gimbal.moveTo(GimbalAxis::Yaw, value);
  } else if (device == DEVICE_PITCH && value <= 180) {
    gimbal.moveTo(GimbalAxis::Pitch, value);
  } else if (device == DEVICE_LASER && value <= 1) {
    digitalWrite(LASER_PIN, value == 1 ? HIGH : LOW);
  } else if (device == DEVICE_STOP && value == 0) {
    gimbal.stop();
  } else if (device == DEVICE_SPEED && value > 0) {
    gimbal.setSpeed(value);
  } else if (device == DEVICE_PING) {
    Serial.println(FIRMWARE_ID);
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    const uint8_t receivedByte = static_cast<uint8_t>(Serial.read());

    switch (parserState) {
      case ParserState::WaitingForHeader:
        if (receivedByte == FRAME_HEADER) {
          parserState = ParserState::WaitingForDevice;
        }
        break;

      case ParserState::WaitingForDevice:
        if (isDevice(receivedByte)) {
          pendingDevice = receivedByte;
          parserState = ParserState::WaitingForValue;
        } else if (receivedByte != FRAME_HEADER) {
          // Invalid device. Discard the partial frame and resynchronize.
          parserState = ParserState::WaitingForHeader;
        }
        // A repeated header keeps us waiting for a valid device byte.
        break;

      case ParserState::WaitingForValue:
        if (isFineDevice(pendingDevice)) {
          pendingHighByte = receivedByte;
          parserState = ParserState::WaitingForLowByte;
        } else {
          applyCommand(pendingDevice, receivedByte);
          parserState = ParserState::WaitingForHeader;
        }
        break;

      case ParserState::WaitingForLowByte:
        applyFineCommand(pendingDevice,
                         static_cast<uint16_t>(pendingHighByte << 8) |
                             receivedByte);
        parserState = ParserState::WaitingForHeader;
        break;
    }
  }
}
}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);

  // Keep the laser off during boot until an explicit ON command is received.
  // Set the output latch before enabling the pin to avoid an ON glitch.
  digitalWrite(LASER_PIN, LOW);
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);

  gimbal.begin();
}

void loop() {
  readSerialCommands();
  gimbal.update();
}
