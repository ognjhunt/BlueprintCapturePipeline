"""Reusable OSCAR prompt and conditioning-mode contract."""

from __future__ import annotations

import hashlib


OSCAR_PUBLIC_SOURCE_REVISION = "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
OSCAR_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with "
    "no motion, motion blur, over-saturation, shaky footage, low resolution, "
    "grainy texture, pixelated images, poorly lit areas, underexposed and "
    "overexposed scenes, poor color balance, washed out colors, choppy "
    "sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)
OSCAR_DEFAULT_NEGATIVE_PROMPT_SHA256 = hashlib.sha256(
    OSCAR_DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
).hexdigest()
FIRST_PERSON_CONDITIONING_MODES = {
    "first_person_review_video",
    "selected_review_video_passthrough",
    "egocentric_review_video_passthrough",
}
EGOCENTRIC_ARM_SKELETON_MODES = {
    "egocentric_arm_skeleton",
    "egocentric_hand_skeleton",
    "first_person_arm_skeleton",
}
TEXTURE_FREE_EGOCENTRIC_ARM_SKELETON_MODES = {
    "texture_free_egocentric_arm_skeleton",
    "oscar_texture_free_egocentric_arm_skeleton",
}
OSCAR_GRIPPER_SCENARIO_PROXY_MODES = {
    "oscar_gripper_scenario_proxy",
    "oscar_egocentric_gripper_proxy",
    "egocentric_rgb_gripper_proxy",
}
PROJECTED_G1_SKELETON_CONDITIONING_MODES = {
    "projected_g1_skeleton",
    "g1_projected_skeleton",
    "unitree_g1_projected_skeleton",
    "projected_g1_arm_hand_skeleton",
    "projected_robot_skeleton",
    "camera_aligned_robot_skeleton",
}
PROJECTED_G1_SKELETON_RGB_OVERLAY_MODES = {
    "projected_g1_skeleton_rgb_overlay",
    "projected_g1_skeleton_scene_overlay",
    "unitree_g1_projected_skeleton_rgb_overlay",
    "projected_robot_skeleton_rgb_overlay",
    "camera_aligned_robot_skeleton_rgb_overlay",
}
ALL_PROJECTED_G1_SKELETON_MODES = (
    PROJECTED_G1_SKELETON_CONDITIONING_MODES
    | PROJECTED_G1_SKELETON_RGB_OVERLAY_MODES
)
PROXY_SKELETON_CONDITIONING_MODES = {
    "scene_overlay_proxy_skeleton",
    "proxy_skeleton",
    "blueprint_proxy_skeleton",
}
