"""Robot configuration CLI helpers for RL-Games scripts."""


ROBOT_CHOICES = ("fr3", "fr3_wc", "fr3_ft_wc")


def add_robot_arg(parser):
    """Add the shared robot-selection argument."""
    parser.add_argument(
        "--robot",
        type=str,
        default="fr3",
        choices=ROBOT_CHOICES,
        help="Robot articulation cfg to use. 'fr3' keeps the task's default robot cfg.",
    )


def apply_robot_cfg(env_cfg, robot_name: str):
    """Override the direct Factory robot articulation config when requested."""
    if robot_name == "fr3":
        return

    if not hasattr(env_cfg, "robot"):
        raise ValueError(f"--robot {robot_name!r} is only supported for env configs with a robot articulation cfg.")

    if robot_name == "fr3_wc":
        from fr3_manipulation.tasks.direct.robots.fr3_wc_cfg import FR3_WC_CFG

        robot_cfg = FR3_WC_CFG
    elif robot_name == "fr3_ft_wc":
        from fr3_manipulation.tasks.direct.robots.fr3_ft_wc_cfg import FR3_FT_WC_CFG

        robot_cfg = FR3_FT_WC_CFG
    else:
        raise ValueError(f"Unsupported robot: {robot_name}")

    env_cfg.robot = robot_cfg.replace(prim_path=env_cfg.robot.prim_path)


def enable_cameras_for_robot(args_cli):
    """Enable camera support when the selected robot includes camera sensors."""
    if args_cli.robot in ("fr3_wc", "fr3_ft_wc"):
        args_cli.enable_cameras = True
