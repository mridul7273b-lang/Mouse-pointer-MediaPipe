""" Mouse Controller - full-frame tracking and stable pinch clicks.

Gestures
--------
1. Index finger extended: move the cursor across the ENTIRE webcam frame,
   mapped to the ENTIRE desktop screen.
2. Thumb + index pinch: left click.
3. Extend index + middle first, then pinch thumb + middle: right click.
4. Close all four fingers: freeze/hold the cursor. Reopen the index finger
   to resume movement.

Press Q in the camera window to quit. PyAutoGUI's emergency failsafe remains
on: move the physical pointer to the upper-left corner to stop the program.
"""

import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# ============================================================
# USER SETTINGS
# ============================================================

# Full-frame mapping: no inner control box or green boundary is used.
# The full width and height of the webcam image map to the full desktop.

# Responsive adaptive smoothing. Increase MIN_CUTOFF for less lag;
# increase BETA for faster response to quick movements.
FILTER_MIN_CUTOFF = 2.0
FILTER_BETA = 0.010
FILTER_D_CUTOFF = 1.0

# Smallest accepted pinch. A larger ratio means the click can happen
# before the fingertips have to touch completely.
LEFT_CLICK_RATIO = 0.55
RIGHT_CLICK_RATIO = 0.55

# Fingers must be apart before a new pinch can be armed.
LEFT_ARM_RATIO = 0.62
RIGHT_ARM_RATIO = 0.62
ARM_STABLE_SECONDS = 0.05

# Lock early, as the selected pair begins closing, before the click.
PRELOCK_MAX_RATIO = 0.92
PRELOCK_CLOSING_DELTA = 0.045
PAIR_SEPARATION_MARGIN = 0.025

# Pinch stability and release hysteresis.
CLICK_CONFIRM_SECONDS = 0.055
PINCH_CANCEL_RATIO = 0.74
PINCH_CANCEL_SECONDS = 0.12
PINCH_LOCK_TIMEOUT = 1.8
RELEASE_RATIO = 0.70
RELEASE_CONFIRM_SECONDS = 0.08

# Simulated physical click duration and repeat protection.
CLICK_HOLD_SECONDS = 0.16
CLICK_COOLDOWN_SECONDS = 0.30

# Fist cursor hold.
FIST_CURL_COSINE = 0.70
FIST_ENTER_COUNT = 4
FIST_KEEP_COUNT = 3
FIST_CONFIRM_SECONDS = 0.12
FIST_RELEASE_SECONDS = 0.12

# Hand pose classification.
EXTENDED_ANGLE_COS = 0.60
EXTENDED_REACH_RATIO = 0.50

# MediaPipe confidence and camera resolution.
DETECTION_CONFIDENCE = 0.70
TRACKING_CONFIDENCE = 0.70
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# PyAutoGUI safety.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


# ============================================================
# ADAPTIVE ONE EURO FILTER
# ============================================================

class OneEuroFilter:
    """Adaptive low-pass filter: smooth when still, responsive when moving."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    @staticmethod
    def _alpha(cutoff, dt):
        cutoff = max(float(cutoff), 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def reset(self, value=None, timestamp=None):
        self.previous_raw = value
        self.previous_filtered = value
        self.previous_derivative = 0.0
        self.previous_time = timestamp

    def __call__(self, value, timestamp):
        value = float(value)
        if self.previous_time is None or self.previous_raw is None:
            self.reset(value, timestamp)
            return value

        dt = min(max(timestamp - self.previous_time, 1.0 / 120.0), 0.1)
        derivative = (value - self.previous_raw) / dt
        derivative_alpha = self._alpha(self.d_cutoff, dt)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.previous_derivative
        )

        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        value_alpha = self._alpha(cutoff, dt)
        filtered = (
            value_alpha * value
            + (1.0 - value_alpha) * self.previous_filtered
        )

        self.previous_raw = value
        self.previous_filtered = filtered
        self.previous_derivative = filtered_derivative
        self.previous_time = timestamp
        return filtered


# ============================================================
# HAND GEOMETRY AND GESTURE CLASSIFICATION
# ============================================================

def landmark_distance(a, b):
    """Euclidean distance in normalized MediaPipe landmark coordinates."""
    return math.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )


def joint_straightness(mcp, pip, dip):
    """Cosine of adjacent bone vectors; close to 1 means a straight finger."""
    v1 = np.array(
        [pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z], dtype=float
    )
    v2 = np.array(
        [dip.x - pip.x, dip.y - pip.y, dip.z - pip.z], dtype=float
    )
    denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denominator < 1e-8:
        return 1.0
    return float(np.dot(v1, v2) / denominator)


def finger_is_extended(lm, ids, palm_width):
    mcp, pip, dip, tip = ids
    straightness = joint_straightness(lm[mcp], lm[pip], lm[dip])
    reach = landmark_distance(lm[mcp], lm[tip]) / max(palm_width, 1e-6)
    return (
        straightness >= EXTENDED_ANGLE_COS
        and reach >= EXTENDED_REACH_RATIO
    )


def get_hand_features(lm):
    palm_width = max(landmark_distance(lm[5], lm[17]), 1e-6)
    finger_ids = (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )
    extended = [finger_is_extended(lm, ids, palm_width) for ids in finger_ids]

    left_ratio = landmark_distance(lm[4], lm[8]) / palm_width
    right_ratio = landmark_distance(lm[4], lm[12]) / palm_width

    return {
        "index_up": extended[0],
        "middle_up": extended[1],
        "curled_count": sum(not is_extended for is_extended in extended),
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
    }


def map_index_to_full_screen(index_tip, screen_width, screen_height):
    """Map the entire webcam image (0..1) to the entire desktop (0..screen)."""
    x = float(np.clip(index_tip.x, 0.0, 1.0))
    y = float(np.clip(index_tip.y, 0.0, 1.0))
    screen_x = x * (screen_width - 1)
    screen_y = y * (screen_height - 1)
    return screen_x, screen_y


def get_click_approach(
    left_ratio,
    right_ratio,
    left_armed,
    right_armed,
    left_baseline,
    right_baseline,
    index_up,
):
    """Return the pinch intentionally approaching the thumb, if any.

    Right click is only eligible after index AND middle were extended and
    separated. Keeping index_up true during the approach prevents an
    index-open/three-fingers-curled pose from being interpreted as right click.
    """
    left_approaching = (
        left_armed
        and left_baseline is not None
        and left_ratio <= PRELOCK_MAX_RATIO
        and (left_baseline - left_ratio) >= PRELOCK_CLOSING_DELTA
        and left_ratio + PAIR_SEPARATION_MARGIN < right_ratio
    )

    right_approaching = (
        right_armed
        and index_up
        and right_baseline is not None
        and right_ratio <= PRELOCK_MAX_RATIO
        and (right_baseline - right_ratio) >= PRELOCK_CLOSING_DELTA
        and right_ratio + PAIR_SEPARATION_MARGIN < left_ratio
    )

    if left_approaching and right_approaching:
        left_progress = left_baseline - left_ratio
        right_progress = right_baseline - right_ratio
        return "left" if left_progress >= right_progress else "right"

    if left_approaching:
        return "left"
    if right_approaching:
        return "right"
    return None


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    cap = cv2.VideoCapture(0)
    hands = None
    mouse_button_down = None

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close other camera apps and try again."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    screen_width, screen_height = pyautogui.size()
    filter_x = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
    filter_y = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)

    # States: MOVE, FIST_PENDING, FIST_HOLD, PINCH_LOCK,
    # CLICK_HELD, WAIT_RELEASE.
    state = "MOVE"
    status = "EXTEND INDEX TO MOVE"

    # Cursor continuity after a pinch/fist: the full frame remains active;
    # this offset only prevents a jump when movement resumes.
    offset_x = 0.0
    offset_y = 0.0
    cursor_initialized = False
    needs_rebase = False

    # Gesture arming: the fingers must first be apart.
    left_armed = False
    right_armed = False
    left_arm_started = None
    right_arm_started = None
    left_baseline = None
    right_baseline = None

    # Click transaction.
    candidate_button = None
    candidate_started = None
    click_confirm_started = None
    cancel_started = None
    anchor_x = 0
    anchor_y = 0
    candidate_cancel_ratio = None

    # Fist hold.
    fist_started = None
    fist_release_started = None

    # Mouse button timing.
    click_release_at = None
    last_click_time = 0.0
    release_started = None

    print("\n========================================")
    print("          AI MOUSE CONTROLLER")
    print("========================================")
    print("INDEX FINGER                  : MOVE")
    print("THUMB + INDEX PINCH           : LEFT CLICK")
    print("INDEX + MIDDLE OPEN, THEN")
    print("THUMB + MIDDLE PINCH          : RIGHT CLICK")
    print("CLOSE ALL FOUR FINGERS        : CURSOR HOLD")
    print("OPEN INDEX                    : RESUME")
    print("FULL WEBCAM FRAME              : FULL DESKTOP")
    print("Q                             : QUIT")
    print("========================================\n")

    try:
        while True:
            now = time.monotonic()

            # Release the mouse button at the end of the deliberate press.
            if (
                state == "CLICK_HELD"
                and click_release_at is not None
                and now >= click_release_at
            ):
                pyautogui.mouseUp(button=mouse_button_down)
                mouse_button_down = None
                click_release_at = None
                state = "WAIT_RELEASE"
                release_started = None
                status = "RELEASE PINCH"

            success, frame = cap.read()
            if not success:
                raise RuntimeError("Webcam frame could not be read.")

            # Mirror the camera for natural hand-to-cursor movement.
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            left_ratio = None
            right_ratio = None
            curled_count = 0
            index_up = False
            middle_up = False
            has_hand = bool(results.multi_hand_landmarks)

            if has_hand:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                features = get_hand_features(lm)
                index_up = features["index_up"]
                middle_up = features["middle_up"]
                curled_count = features["curled_count"]
                left_ratio = features["left_ratio"]
                right_ratio = features["right_ratio"]

                index_tip = lm[8]
                raw_x, raw_y = map_index_to_full_screen(
                    index_tip, screen_width, screen_height
                )

                # ----------------------------------------------------
                # ARM CLICK GESTURES ONLY IN NORMAL MOVEMENT.
                # Left: index is extended and thumb-index are apart.
                # Right: index AND middle are extended and thumb-middle
                # are apart. An index-open/three-curled hand cannot arm right.
                # ----------------------------------------------------
                if state == "MOVE":
                    if index_up and left_ratio >= LEFT_ARM_RATIO:
                        if left_arm_started is None:
                            left_arm_started = now
                        elif now - left_arm_started >= ARM_STABLE_SECONDS:
                            left_armed = True
                            left_baseline = max(left_baseline or 0.0, left_ratio)
                    elif not left_armed:
                        left_arm_started = None

                    if index_up and middle_up and right_ratio >= RIGHT_ARM_RATIO:
                        if right_arm_started is None:
                            right_arm_started = now
                        elif now - right_arm_started >= ARM_STABLE_SECONDS:
                            right_armed = True
                            right_baseline = max(right_baseline or 0.0, right_ratio)
                    elif not right_armed:
                        right_arm_started = None

                    # Losing the index extension disarms right-click immediately.
                    if not index_up:
                        right_armed = False
                        right_arm_started = None
                        right_baseline = None

                # ----------------------------------------------------
                # STATE: MOVE
                # ----------------------------------------------------
                if state == "MOVE":
                    approach = get_click_approach(
                        left_ratio,
                        right_ratio,
                        left_armed,
                        right_armed,
                        left_baseline,
                        right_baseline,
                        index_up,
                    )

                    # Detect a pinch transition BEFORE the fingertips touch.
                    # Save the real desktop position and stop sending cursor
                    # movement commands until the click/cancel is complete.
                    if approach is not None and now - last_click_time >= CLICK_COOLDOWN_SECONDS:
                        current = pyautogui.position()
                        anchor_x, anchor_y = current.x, current.y
                        candidate_button = approach
                        candidate_started = now
                        click_confirm_started = None
                        cancel_started = None
                        selected_baseline = (
                            left_baseline if approach == "left" else right_baseline
                        )
                        selected_start_ratio = (
                            left_ratio if approach == "left" else right_ratio
                        )
                        candidate_cancel_ratio = min(
                            selected_baseline - 0.02,
                            selected_start_ratio + 0.06,
                        )
                        candidate_cancel_ratio = max(
                            PINCH_CANCEL_RATIO, candidate_cancel_ratio
                        )
                        state = "PINCH_LOCK"
                        left_armed = right_armed = False
                        left_arm_started = right_arm_started = None
                        left_baseline = right_baseline = None
                        status = f"{approach.upper()} — POINTER LOCKED"

                    # Fist hold: four curled fingers; an index-up pose is never
                    # treated as a fist. This check follows pinch approach.
                    elif curled_count >= FIST_ENTER_COUNT and not index_up:
                        current = pyautogui.position()
                        anchor_x, anchor_y = current.x, current.y
                        fist_started = now
                        fist_release_started = None
                        state = "FIST_PENDING"
                        left_armed = right_armed = False
                        left_arm_started = right_arm_started = None
                        left_baseline = right_baseline = None
                        status = "CONFIRMING FIST — POINTER LOCKED"

                    elif index_up:
                        if not cursor_initialized:
                            # First tracking frame maps directly across the full
                            # camera frame to the full desktop (no initial offset).
                            cursor_initialized = True
                            offset_x = 0.0
                            offset_y = 0.0
                            filter_x.reset(raw_x, now)
                            filter_y.reset(raw_y, now)
                            needs_rebase = False
                            pyautogui.moveTo(
                                int(np.clip(round(raw_x), 0, screen_width - 1)),
                                int(np.clip(round(raw_y), 0, screen_height - 1)),
                            )
                        elif needs_rebase:
                            # After a click/hold, preserve the pointer position
                            # for the first resumed frame so it does not jump.
                            current = pyautogui.position()
                            offset_x = current.x - raw_x
                            offset_y = current.y - raw_y
                            filter_x.reset(current.x, now)
                            filter_y.reset(current.y, now)
                            needs_rebase = False
                        else:
                            target_x = float(np.clip(raw_x + offset_x, 0, screen_width - 1))
                            target_y = float(np.clip(raw_y + offset_y, 0, screen_height - 1))
                            cursor_x = filter_x(target_x, now)
                            cursor_y = filter_y(target_y, now)
                            pyautogui.moveTo(
                                int(np.clip(round(cursor_x), 0, screen_width - 1)),
                                int(np.clip(round(cursor_y), 0, screen_height - 1)),
                            )
                        status = "CURSOR MOVEMENT — FULL SCREEN"
                    else:
                        status = "EXTEND INDEX TO MOVE"

                # ----------------------------------------------------
                # STATE: FIST ENTRY CONFIRMATION
                # ----------------------------------------------------
                elif state == "FIST_PENDING":
                    if index_up or curled_count < FIST_KEEP_COUNT:
                        state = "MOVE"
                        fist_started = None
                        needs_rebase = True
                        status = "FIST CANCELLED"
                    elif now - fist_started >= FIST_CONFIRM_SECONDS:
                        state = "FIST_HOLD"
                        fist_release_started = None
                        status = "CURSOR HOLD ACTIVE"
                    else:
                        status = "CONFIRMING FIST"

                # ----------------------------------------------------
                # STATE: CLOSED FIST HOLD
                # ----------------------------------------------------
                elif state == "FIST_HOLD":
                    # Opening the index releases the cursor after a short,
                    # stable confirmation. Other three curled fingers are OK.
                    if index_up:
                        if fist_release_started is None:
                            fist_release_started = now
                        elif now - fist_release_started >= FIST_RELEASE_SECONDS:
                            state = "MOVE"
                            fist_started = None
                            fist_release_started = None
                            needs_rebase = True
                            status = "CURSOR RESUMED"
                        else:
                            status = "OPENING INDEX — RESUMING"
                    else:
                        fist_release_started = None
                        status = "CURSOR HOLD ACTIVE"

                # ----------------------------------------------------
                # STATE: PINCH LOCK / CLICK CONFIRMATION
                # ----------------------------------------------------
                elif state == "PINCH_LOCK":
                    selected_ratio = left_ratio if candidate_button == "left" else right_ratio
                    other_ratio = right_ratio if candidate_button == "left" else left_ratio
                    click_ratio = LEFT_CLICK_RATIO if candidate_button == "left" else RIGHT_CLICK_RATIO

                    # No moveTo calls occur while locked. This is important:
                    # the pointer cannot jitter as the pinching finger moves.
                    valid_pose = candidate_button != "right" or index_up
                    correct_pair = selected_ratio + PAIR_SEPARATION_MARGIN < other_ratio
                    returned_open = selected_ratio >= candidate_cancel_ratio
                    timed_out = now - candidate_started >= PINCH_LOCK_TIMEOUT
                    should_cancel = (not valid_pose) or returned_open or timed_out

                    if should_cancel:
                        if cancel_started is None:
                            cancel_started = now
                        elif now - cancel_started >= PINCH_CANCEL_SECONDS:
                            state = "MOVE"
                            candidate_button = None
                            candidate_started = None
                            click_confirm_started = None
                            cancel_started = None
                            candidate_cancel_ratio = None
                            needs_rebase = True
                            status = "PINCH CANCELLED — POINTER UNLOCKED"
                    elif not correct_pair:
                        click_confirm_started = None
                        status = "WRONG FINGER PAIR — KEEP PINCHING SELECTED PAIR"
                    else:
                        cancel_started = None
                        if selected_ratio <= click_ratio:
                            if click_confirm_started is None:
                                click_confirm_started = now
                            elif now - click_confirm_started >= CLICK_CONFIRM_SECONDS:
                                # Restore saved position once, then press. The
                                # cursor remains still until the release state.
                                pyautogui.moveTo(anchor_x, anchor_y)
                                pyautogui.mouseDown(button=candidate_button)
                                mouse_button_down = candidate_button
                                click_release_at = now + CLICK_HOLD_SECONDS
                                last_click_time = now
                                state = "CLICK_HELD"
                                status = f"{candidate_button.upper()} CLICK"
                        else:
                            click_confirm_started = None
                            status = "POINTER LOCKED — SMALL PINCH TO CLICK"

                # ----------------------------------------------------
                # STATE: MOUSE BUTTON HELD
                # ----------------------------------------------------
                elif state == "CLICK_HELD":
                    status = f"{candidate_button.upper()} CLICK — PRESS"

                # ----------------------------------------------------
                # STATE: WAIT FOR PINCH RELEASE
                # ----------------------------------------------------
                elif state == "WAIT_RELEASE":
                    selected_ratio = left_ratio if candidate_button == "left" else right_ratio
                    released = selected_ratio >= RELEASE_RATIO

                    if released:
                        if release_started is None:
                            release_started = now
                        elif now - release_started >= RELEASE_CONFIRM_SECONDS:
                            state = "MOVE"
                            candidate_button = None
                            candidate_started = None
                            click_confirm_started = None
                            candidate_cancel_ratio = None
                            release_started = None
                            needs_rebase = True
                            left_armed = right_armed = False
                            left_arm_started = right_arm_started = None
                            left_baseline = right_baseline = None
                            status = "READY"
                    else:
                        release_started = None
                        status = "RELEASE PINCH"

                # Draw only the index fingertip marker. No green control box:
                # the whole webcam image is the active movement surface.
                ix = int(np.clip(lm[8].x * frame_width, 0, frame_width - 1))
                iy = int(np.clip(lm[8].y * frame_height, 0, frame_height - 1))
                cv2.circle(frame, (ix, iy), 7, (255, 0, 255), -1)

            else:
                # No hand: clear stale arming data. Do not infer a gesture
                # from the previous frame's landmarks.
                if state == "MOVE":
                    needs_rebase = True
                    left_armed = right_armed = False
                    left_arm_started = right_arm_started = None
                    left_baseline = right_baseline = None
                    status = "NO HAND — SHOW INDEX"
                elif state == "FIST_PENDING":
                    state = "MOVE"
                    fist_started = None
                    needs_rebase = True
                    status = "FIST CANCELLED"
                elif state == "FIST_HOLD":
                    status = "CURSOR HOLD — SHOW OPEN INDEX TO RESUME"
                elif state == "PINCH_LOCK":
                    state = "MOVE"
                    candidate_button = None
                    candidate_started = None
                    click_confirm_started = None
                    cancel_started = None
                    candidate_cancel_ratio = None
                    needs_rebase = True
                    status = "PINCH CANCELLED — HAND LOST"
                elif state == "CLICK_HELD":
                    status = "CLICK IN PROGRESS"
                elif state == "WAIT_RELEASE":
                    status = "WAITING FOR PINCH RELEASE"

            # --------------------------------------------------------
            # CAMERA OVERLAY — NO INNER CONTROL RECTANGLE
            # --------------------------------------------------------
            cv2.rectangle(frame, (0, 0), (frame_width, 108), (25, 25, 25), -1)
            cv2.putText(
                frame, "ACHE LAG RHE HO HE!HE!HE!", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"STATUS: {status}", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2
            )
            left_text = "--" if left_ratio is None else f"{left_ratio:.2f}"
            right_text = "--" if right_ratio is None else f"{right_ratio:.2f}"
            cv2.putText(
                frame,
                f"LEFT PINCH: {left_text}   RIGHT PINCH: {right_text}   FINGERS CURLED: {curled_count}/4",
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
            )
            cv2.putText(
                frame,
                "INDEX: MOVE | THUMB+INDEX: LEFT | INDEX+MIDDLE OPEN THEN THUMB+MIDDLE: RIGHT | FIST: HOLD | Q: QUIT",
                (8, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1
            )

            cv2.imshow("Mouse Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if mouse_button_down is not None:
            try:
                pyautogui.mouseUp(button=mouse_button_down)
            except Exception:
                pass
        cap.release()
        if hands is not None:
            hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except pyautogui.FailSafeException:
        print("PyAutoGUI emergency stop activated (top-left corner).")
    except KeyboardInterrupt:
        print("Program interrupted.")
    except Exception as exc:
        print(f"AI Mouse Controller error: {exc}")
    finally:
        print("AI Mouse Controller stopped.")
