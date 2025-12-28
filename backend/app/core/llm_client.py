import httpx
from typing import Optional
import json

from app.config import settings


BETA_PROMPT_TEMPLATE = """你是一位专业的攀岩教练，请根据以下视频分析数据，为攀岩者提供改进建议。

## 分析数据摘要
- 攀爬时长: {duration:.1f}秒
- 平均稳定性评分: {avg_stability:.1f}%
- 动作效率: {efficiency:.1f}%
- 动态爆发次数: {dyno_count}次
- 最大速度: {max_velocity:.2f}
- 最大加速度: {max_acceleration:.2f}

## 关节角度统计
{joint_angles_text}

## 检测到的问题
{detected_issues}

## 请提供以下建议:
1. **重心控制**: 针对重心偏移的改进方法
2. **发力技巧**: 如何更好地利用身体各部位
3. **动作优化**: 具体的动作调整建议
4. **训练建议**: 针对性的训练方法

请用简洁专业的语言回答，每个部分2-3句话。"""


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
        timeout: int = settings.llm_timeout,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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

    avg_stability = summary.get("avg_stability_score", 0) * 100
    if avg_stability < 50:
        issues.append("- 稳定性较低，重心控制需要加强")
    elif avg_stability < 70:
        issues.append("- 稳定性一般，可以进一步优化重心位置")

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


async def generate_beta_suggestion(
    summary: dict,
    client: Optional[OllamaClient] = None,
) -> Optional[str]:
    """
    Generate beta suggestion using LLM.

    Args:
        summary: Analysis summary dictionary
        client: Optional OllamaClient instance

    Returns:
        Beta suggestion text or None if failed
    """
    if client is None:
        client = OllamaClient()

    # Check if LLM is available
    if not await client.is_available():
        return generate_fallback_suggestion(summary)

    # Format the prompt
    joint_angles_text = format_joint_angles(summary.get("joint_angle_stats", {}))
    detected_issues = detect_issues(summary)

    prompt = BETA_PROMPT_TEMPLATE.format(
        duration=summary.get("duration", 0),
        avg_stability=summary.get("avg_stability_score", 0) * 100,
        efficiency=summary.get("avg_efficiency", 0) * 100,
        dyno_count=summary.get("dyno_count", 0),
        max_velocity=summary.get("max_velocity", 0),
        max_acceleration=summary.get("max_acceleration", 0),
        joint_angles_text=joint_angles_text,
        detected_issues=detected_issues,
    )

    suggestion = await client.generate(prompt)

    if suggestion is None:
        return generate_fallback_suggestion(summary)

    return suggestion


def generate_fallback_suggestion(summary: dict) -> str:
    """Generate a basic suggestion when LLM is not available."""
    suggestions = []

    avg_stability = summary.get("avg_stability_score", 0) * 100

    if avg_stability < 50:
        suggestions.append("**重心控制**: 建议加强核心力量训练，注意将重心保持在支撑点范围内。")
    elif avg_stability < 70:
        suggestions.append("**重心控制**: 重心控制尚可，可以通过练习静态动作来进一步提高稳定性。")
    else:
        suggestions.append("**重心控制**: 重心控制良好，继续保持当前的身体姿态意识。")

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
