import httpx
from typing import Optional
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def get_current_model() -> str:
    """Get the currently configured model from runtime config."""
    try:
        from app.api.routes.llm import get_current_model as _get_model
        return _get_model()
    except ImportError:
        return settings.ollama_model


BETA_PROMPT_TEMPLATE = """你是一位专业的攀岩教练，请根据以下视频分析数据，为攀岩者提供改进建议。

## 分析数据摘要
- 攀爬时长: {duration:.1f}秒
- 动作效率: {efficiency:.1f}%
- 动态爆发次数: {dyno_count}次
- 最大速度: {max_velocity:.2f}
- 最大加速度: {max_acceleration:.2f}

## 关节角度统计
{joint_angles_text}

## 检测到的攀岩动作
{movements_text}

## 检测到的问题
{detected_issues}

## 请提供以下建议:
1. **重心控制**: 针对重心偏移的改进方法
2. **发力技巧**: 如何更好地利用身体各部位
3. **动作优化**: 针对检测到的具体动作，给出改进建议
4. **训练建议**: 针对性的训练方法

请用简洁专业的语言回答，每个部分2-3句话。如果检测到特定的攀岩动作（如跟勾、侧拉等），请针对这些动作给出专业建议。"""


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: Optional[str] = None,
        timeout: int = settings.llm_timeout,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model or get_current_model()
        self.timeout = timeout

    async def list_models(self) -> list[dict]:
        """List all available models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except Exception:
            return []

    async def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text completion using Ollama.

        Args:
            prompt: Input prompt

        Returns:
            Generated text or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response")
        except httpx.TimeoutException:
            return None
        except httpx.HTTPError:
            return None
        except Exception:
            return None

    async def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


def format_joint_angles(joint_angle_stats: dict) -> str:
    """Format joint angle statistics for the prompt."""
    lines = []
    name_map = {
        "left_elbow": "左肘",
        "right_elbow": "右肘",
        "left_shoulder": "左肩",
        "right_shoulder": "右肩",
        "left_hip": "左髋",
        "right_hip": "右髋",
        "left_knee": "左膝",
        "right_knee": "右膝",
    }

    for joint, stats in joint_angle_stats.items():
        if stats:
            cn_name = name_map.get(joint, joint)
            lines.append(f"- {cn_name}: 平均 {stats['avg']:.1f}° (范围: {stats['min']:.1f}° - {stats['max']:.1f}°)")

    return "\n".join(lines) if lines else "- 无有效数据"


def detect_issues(summary: dict) -> str:
    """Detect potential issues from the analysis summary."""
    issues = []

    efficiency = summary.get("avg_efficiency", 0) * 100
    if efficiency < 50:
        issues.append("- 动作效率较低，路线选择或动作序列需要优化")
    elif efficiency < 70:
        issues.append("- 动作效率中等，存在优化空间")

    # Check joint angles
    joint_stats = summary.get("joint_angle_stats", {})

    # Check for locked arms (elbow angle > 160)
    for side in ["left", "right"]:
        elbow = joint_stats.get(f"{side}_elbow", {})
        if elbow.get("avg", 0) > 160:
            issues.append(f"- {'左' if side == 'left' else '右'}臂过直，应保持微曲以减少疲劳")

    # Check for high shoulders
    for side in ["left", "right"]:
        shoulder = joint_stats.get(f"{side}_shoulder", {})
        if shoulder.get("avg", 0) < 60:
            issues.append(f"- {'左' if side == 'left' else '右'}肩角度较小，可能存在耸肩问题")

    if not issues:
        issues.append("- 整体表现良好，继续保持")

    return "\n".join(issues)


def format_movements(movements: list[dict]) -> str:
    """Format detected movements for the prompt."""
    if not movements:
        return "- 未检测到特定攀岩动作"

    lines = []

    # Group movements by type
    by_type: dict[str, list[dict]] = {}
    for m in movements:
        type_name = m.get("movement_name_cn", m.get("movement_type", "未知"))
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(m)

    for type_name, type_movements in by_type.items():
        count = len(type_movements)
        challenging_count = sum(1 for m in type_movements if m.get("is_challenging", False))

        # Get representative movement details
        sample = type_movements[0]
        side_cn = sample.get("side_cn", "")
        key_angles = sample.get("key_angles", {})

        # Build description
        desc_parts = [f"- {type_name}: {count}次"]
        if side_cn and side_cn != "双侧":
            desc_parts.append(f"({side_cn}为主)")
        if challenging_count > 0:
            desc_parts.append(f"，其中{challenging_count}次为高难度")

        # Add key angle info if available
        angle_info = []
        if "elbow" in key_angles:
            angle_info.append(f"肘部角度约{key_angles['elbow']:.0f}°")
        if "knee" in key_angles:
            angle_info.append(f"膝部角度约{key_angles['knee']:.0f}°")
        if "hip" in key_angles:
            angle_info.append(f"髋部角度约{key_angles['hip']:.0f}°")

        if angle_info:
            desc_parts.append(f"，{', '.join(angle_info)}")

        lines.append("".join(desc_parts))

    # Add summary
    total = len(movements)
    total_challenging = sum(1 for m in movements if m.get("is_challenging", False))
    lines.append(f"- 总计: {total}个技术动作")
    if total_challenging > 0:
        lines.append(f"- 高难度动作: {total_challenging}次")

    return "\n".join(lines)


async def generate_beta_suggestion(
    summary: dict,
    movements: Optional[list[dict]] = None,
    client: Optional[OllamaClient] = None,
) -> Optional[str]:
    """
    Generate beta suggestion using LLM.

    Args:
        summary: Analysis summary dictionary
        movements: Optional list of detected movements from movement detector
        client: Optional OllamaClient instance

    Returns:
        Beta suggestion text or None if failed
    """
    if client is None:
        client = OllamaClient()

    # Format the prompt first (for logging)
    joint_angles_text = format_joint_angles(summary.get("joint_angle_stats", {}))
    movements_text = format_movements(movements or [])
    detected_issues = detect_issues(summary)

    prompt = BETA_PROMPT_TEMPLATE.format(
        duration=summary.get("duration", 0),
        efficiency=summary.get("avg_efficiency", 0) * 100,
        dyno_count=summary.get("dyno_count", 0),
        max_velocity=summary.get("max_velocity", 0),
        max_acceleration=summary.get("max_acceleration", 0),
        joint_angles_text=joint_angles_text,
        movements_text=movements_text,
        detected_issues=detected_issues,
    )

    logger.info(f"Beta suggestion prompt:\n{prompt}")

    # Check if LLM is available
    if not await client.is_available():
        logger.warning(f"Ollama not available at {client.base_url}, using fallback suggestion")
        return generate_fallback_suggestion(summary)

    logger.info(f"Sending prompt to Ollama model: {client.model}")
    suggestion = await client.generate(prompt)

    if suggestion is None:
        return generate_fallback_suggestion(summary)

    return suggestion


def generate_fallback_suggestion(summary: dict) -> str:
    """Generate a basic suggestion when LLM is not available."""
    suggestions = []

    suggestions.append("**重心控制**: 注意将重心保持在支撑点范围内，通过练习静态动作来提高稳定性。")

    efficiency = summary.get("avg_efficiency", 0) * 100
    if efficiency < 60:
        suggestions.append("**动作优化**: 尝试减少不必要的身体摆动，规划更直接的移动路线。")
    else:
        suggestions.append("**动作优化**: 动作效率良好，可以尝试挑战更复杂的路线。")

    dyno_count = summary.get("dyno_count", 0)
    if dyno_count > 0:
        suggestions.append(f"**发力技巧**: 检测到{dyno_count}次动态动作，注意起跳时机和落点精准度。")
    else:
        suggestions.append("**发力技巧**: 当前以静态动作为主，可以适当练习动态技巧以应对更多路线类型。")

    suggestions.append("**训练建议**: 建议进行针对性的核心力量和指力训练，每周保持2-3次攀爬练习。")

    return "\n\n".join(suggestions)


# Movement description prompt template
MOVEMENT_DESCRIPTION_PROMPT_TEMPLATE = """你是一位专业的攀岩教练。请为检测到的攀岩动作生成简洁的中文描述。

## 检测到的动作
- 动作类型: {movement_type_cn}
- 持续时间: {duration:.1f}秒
- 使用侧: {side_cn}
- 难度等级: {difficulty}

## 关键角度数据
{angles_text}

## 请生成描述
要求:
1. 简洁描述动作特点和执行质量（1-2句话）
2. 如果是高难度动作，突出其挑战性
3. 使用专业但易懂的语言

示例格式:
"侧拉: 左侧拉动，肘部角度75°，保持良好的身体张力。"

请直接输出描述，不要包含其他内容。"""


def format_movement_angles(key_angles: dict) -> str:
    """Format movement angles for the prompt."""
    angle_names = {
        "elbow": "肘部角度",
        "shoulder": "肩部角度",
        "hip": "髋部角度",
        "knee": "膝部角度",
        "h_displacement": "水平位移",
        "extension": "延伸距离",
        "acceleration": "加速度",
        "velocity": "速度",
    }

    lines = []
    for key, value in key_angles.items():
        cn_name = angle_names.get(key, key)
        if isinstance(value, float):
            if key in ["h_displacement", "extension"]:
                lines.append(f"- {cn_name}: {value:.2f}")
            else:
                lines.append(f"- {cn_name}: {value:.1f}°")
        else:
            lines.append(f"- {cn_name}: {value}")

    return "\n".join(lines) if lines else "- 无详细数据"


def generate_movement_fallback_description(movement: dict) -> str:
    """Generate a basic movement description without LLM."""
    movement_type_cn = movement.get("movement_name_cn", "未知动作")
    side_cn = movement.get("side_cn", "")
    duration = movement.get("end_timestamp", 0) - movement.get("start_timestamp", 0)
    is_challenging = movement.get("is_challenging", False)
    key_angles = movement.get("key_angles", {})

    # Build description based on movement type
    difficulty_text = "高难度" if is_challenging else ""

    # Get primary angle info
    angle_info = ""
    if "elbow" in key_angles:
        angle_info = f"肘部角度{key_angles['elbow']:.0f}°"
    elif "knee" in key_angles:
        angle_info = f"膝部角度{key_angles['knee']:.0f}°"
    elif "hip" in key_angles:
        angle_info = f"髋部角度{key_angles['hip']:.0f}°"

    if angle_info:
        return f"{difficulty_text}{movement_type_cn}: {side_cn}执行，{angle_info}，持续{duration:.1f}秒。"
    else:
        return f"{difficulty_text}{movement_type_cn}: {side_cn}执行，持续{duration:.1f}秒。"


async def generate_movement_description(
    movement: dict,
    client: Optional[OllamaClient] = None,
) -> str:
    """
    Generate Chinese description for a single movement using LLM.

    Args:
        movement: Movement dictionary with type, angles, etc.
        client: Optional OllamaClient instance

    Returns:
        Generated description text
    """
    if client is None:
        client = OllamaClient()

    # Check if LLM is available
    if not await client.is_available():
        return generate_movement_fallback_description(movement)

    # Format the prompt
    movement_type_cn = movement.get("movement_name_cn", "未知动作")
    side_cn = movement.get("side_cn", "")
    duration = movement.get("end_timestamp", 0) - movement.get("start_timestamp", 0)
    is_challenging = movement.get("is_challenging", False)
    key_angles = movement.get("key_angles", {})

    difficulty = "高难度动作" if is_challenging else "标准动作"
    angles_text = format_movement_angles(key_angles)

    prompt = MOVEMENT_DESCRIPTION_PROMPT_TEMPLATE.format(
        movement_type_cn=movement_type_cn,
        duration=duration,
        side_cn=side_cn,
        difficulty=difficulty,
        angles_text=angles_text,
    )

    description = await client.generate(prompt)

    if description is None:
        return generate_movement_fallback_description(movement)

    # Clean up the response (remove extra whitespace/newlines)
    return description.strip()


async def generate_movement_descriptions(
    movements: list[dict],
    client: Optional[OllamaClient] = None,
    use_llm: bool = True,
) -> list[dict]:
    """
    Generate Chinese descriptions for all movements.

    Args:
        movements: List of movement dictionaries
        client: Optional OllamaClient instance
        use_llm: Whether to use LLM (False = use fallback only)

    Returns:
        List of movements with description_cn field populated
    """
    if client is None:
        client = OllamaClient()

    # Check LLM availability once
    llm_available = use_llm and await client.is_available()

    if llm_available:
        logger.info(f"Generating LLM descriptions for {len(movements)} movements")
    else:
        logger.info(f"Using fallback descriptions for {len(movements)} movements")

    for movement in movements:
        if llm_available:
            description = await generate_movement_description(movement, client)
        else:
            description = generate_movement_fallback_description(movement)

        movement["description_cn"] = description

    return movements
