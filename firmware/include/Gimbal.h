#pragma once

#include <Arduino.h>
#include <ESP32PWM.h>

enum class GimbalAxis : uint8_t {
  Yaw,
  Pitch,
};

struct GimbalServoConfig {
  int pin;
  float minimumAngle;
  float maximumAngle;
  float startupAngle;
  int minimumPulseUs;
  int maximumPulseUs;
};

class Gimbal {
 public:
  Gimbal(const GimbalServoConfig& yawConfig,
         const GimbalServoConfig& pitchConfig,
         float speedDegreesPerSecond = 60.0f);

  void begin();

  // Sets a target and returns immediately. Call update() on every loop pass.
  void moveTo(GimbalAxis axis, float angle);
  void moveTo(float yawAngle, float pitchAngle);

  // Moves an axis immediately instead of applying the configured speed.
  void moveImmediately(GimbalAxis axis, float angle);

  void update();
  // Cancel pending travel and hold the most recently written servo positions.
  void stop();
  void setSpeed(float degreesPerSecond);

  float currentAngle(GimbalAxis axis) const;
  float targetAngle(GimbalAxis axis) const;
  bool isMoving(GimbalAxis axis) const;

 private:
  struct AxisState {
    float currentAngle;
    float targetAngle;
  };

  static constexpr unsigned long UPDATE_INTERVAL_MS = 20;
  static constexpr double SERVO_FREQUENCY_HZ = 50.0;
  static constexpr float SERVO_PERIOD_US = 20000.0f;
  // About 1.2 us (0.11 degrees) per step at 50 Hz. The Servo class is stuck
  // at 10 bits (19.5 us, 1.76 degrees): its setTimerWidth() re-attaches the
  // pin, which on the ESP32-S3 puts both servos on the same MCPWM output.
  static constexpr uint8_t PWM_RESOLUTION_BITS = 14;

  ESP32PWM yawPwm_;
  ESP32PWM pitchPwm_;
  GimbalServoConfig yawConfig_;
  GimbalServoConfig pitchConfig_;
  AxisState yawState_;
  AxisState pitchState_;
  float speedDegreesPerSecond_;
  unsigned long lastUpdateMs_;

  AxisState& stateFor(GimbalAxis axis);
  const AxisState& stateFor(GimbalAxis axis) const;
  const GimbalServoConfig& configFor(GimbalAxis axis) const;
  ESP32PWM& pwmFor(GimbalAxis axis);
  void updateAxis(GimbalAxis axis, float maximumStep);
  // Writes a fractional angle as a pulse width instead of whole degrees.
  void writeAngle(GimbalAxis axis, float angle);
};
