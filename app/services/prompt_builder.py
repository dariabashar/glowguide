from app.schemas import MakeupSpec

def build_prompt_from_spec(spec: MakeupSpec) -> str:
    return (
        f"A close-up portrait of a young woman with {spec.foundation.tone} skin and {spec.foundation.undertone} undertone. "
        f"Apply makeup using: {spec.foundation.coverage} coverage foundation with a smooth, soft-matte finish. "
        f"Place {spec.blush.color} blush ({spec.blush.finish}) on the {spec.blush.placement}. "
        f"Use {spec.eyes.shadow_color} eyeshadow, {spec.eyes.liner_style} eyeliner, and "
        f"{'mascara to define lashes' if spec.eyes.mascara else 'no mascara'}. "
        f"Lips in {spec.lips.color} with a {spec.lips.finish} finish. "
        f"The makeup should be natural yet visible, elegant, and refined. "
        f"Use the same face, pose, lighting, and background. Do not generate a different identity or style. "
        f"Avoid exaggerated features or cartoonish appearance. Generate a single realistic portrait."
    )

