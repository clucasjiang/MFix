#include "Gimbal.h"

#include <math.h>

Gimbal::Gimbal(const GimbalServoConfig& yawConfig,
               const GimbalServoConfig& pitchConfig,
               float speedDegreesPerSecond)
    : yawConfig_(yawConfig),
      pitchConfig_(pitchConfig),
      yawState_{yawConfig.startupAngle, yawConfig.startupAngle},
      pitchState_{pitchConfig.startupAngle, pitchConfig.startupAngle},
      speedDegreesPerSecond_(speedDegreesPerSecond),
      lastUpdateMs_(0) {}

void Gimbal::begin() {
  // Attach each pin once, at full resolution. Each gets its own PWM output.
  yawPwm_.attachPin(yawConfig_.pin, SERVO_FREQUENCY_HZ, PWM_RESOLUTION_BITS);
  pitchPwm_.attachPin(pitchConfig_.pin, SERVO_FREQUENCY_HZ,
                      PWM_RESOLUTION_BITS);

  moveImmediately(GimbalAxis::Yaw, yawConfig_.startupAngle);
  moveImmediately(GimbalAxis::Pitch, pitchConfig_.startupAngle);
  lastUpdateMs_ = millis();
}

void Gimbal::moveTo(GimbalAxis axis, float angle) {
  const GimbalServoConfig& config = configFor(axis);
  stateFor(axis).targetAngle =
      constrain(angle, config.minimumAngle, config.maximumAngle);
}

void Gimbal::moveTo(float yawAngle, float pitchAngle) {
  moveTo(GimbalAxis::Yaw, yawAngle);
  moveTo(GimbalAxis::Pitch, pitchAngle);
}

void Gimbal::moveImmediately(GimbalAxis axis, float angle) {
  const GimbalServoConfig& config = configFor(axis);
  const float limitedAngle =
      constrain(angle, config.minimumAngle, config.maximumAngle);
  AxisState& state = stateFor(axis);

  state.currentAngle = limitedAngle;
  state.targetAngle = limitedAngle;
  writeAngle(axis, limitedAngle);
}

void Gimbal::update() {
  const unsigned long now = millis();
  const unsigned long elapsedMs = now - lastUpdateMs_;

  if (elapsedMs < UPDATE_INTERVAL_MS) {
    return;
  }

  lastUpdateMs_ = now;
  const float maximumStep =
      speedDegreesPerSecond_ * static_cast<float>(elapsedMs) / 1000.0f;

  updateAxis(GimbalAxis::Yaw, maximumStep);
  updateAxis(GimbalAxis::Pitch, maximumStep);
}

void Gimbal::stop() {
  yawState_.targetAngle = yawState_.currentAngle;
  pitchState_.targetAngle = pitchState_.currentAngle;
  lastUpdateMs_ = millis();
}

void Gimbal::setSpeed(float degreesPerSecond) {
  if (degreesPerSecond > 0.0f) {
    speedDegreesPerSecond_ = degreesPerSecond;
  }
}

float Gimbal::currentAngle(GimbalAxis axis) const {
  return stateFor(axis).currentAngle;
}

float Gimbal::targetAngle(GimbalAxis axis) const {
  return stateFor(axis).targetAngle;
}

bool Gimbal::isMoving(GimbalAxis axis) const {
  const AxisState& state = stateFor(axis);
  return fabsf(state.targetAngle - state.currentAngle) > 0.01f;
}

Gimbal::AxisState& Gimbal::stateFor(GimbalAxis axis) {
  return axis == GimbalAxis::Yaw ? yawState_ : pitchState_;
}

const Gimbal::AxisState& Gimbal::stateFor(GimbalAxis axis) const {
  return axis == GimbalAxis::Yaw ? yawState_ : pitchState_;
}

const GimbalServoConfig& Gimbal::configFor(GimbalAxis axis) const {
  return axis == GimbalAxis::Yaw ? yawConfig_ : pitchConfig_;
}

ESP32PWM& Gimbal::pwmFor(GimbalAxis axis) {
  return axis == GimbalAxis::Yaw ? yawPwm_ : pitchPwm_;
}

void Gimbal::updateAxis(GimbalAxis axis, float maximumStep) {
  AxisState& state = stateFor(axis);
  const float remaining = state.targetAngle - state.currentAngle;
  if (fabsf(remaining) <= maximumStep) {
    state.currentAngle = state.targetAngle;
  } else {
    state.currentAngle += remaining > 0.0f ? maximumStep : -maximumStep;
  }

  writeAngle(axis, state.currentAngle);
}

void Gimbal::writeAngle(GimbalAxis axis, float angle) {
  const GimbalServoConfig& config = configFor(axis);
  // Same linear 0-180 degree mapping as Servo::write(), without rounding the
  // angle to a whole degree first.
  const float pulseRangeUs =
      static_cast<float>(config.maximumPulseUs - config.minimumPulseUs);
  const float pulseUs =
      constrain(config.minimumPulseUs + angle * pulseRangeUs / 180.0f,
                static_cast<float>(config.minimumPulseUs),
                static_cast<float>(config.maximumPulseUs));
  const float steps = static_cast<float>(1UL << PWM_RESOLUTION_BITS);
  pwmFor(axis).write(static_cast<uint32_t>(pulseUs * steps / SERVO_PERIOD_US + 0.5f));
}
