"""Smoke test for GradientGuidedSegmenter (B2.1) — updated for new architecture."""
import sys, traceback

def step(msg):
    print(msg, end=" ", flush=True)

try:
    step("Importing...")
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import numpy as np
    print("OK")

    step("Loading model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("OK")

    step("Loading image...")
    input_text = "black dog next to a yellow hydrant"
    input_image = Image.open("assets/dog_and_hydrant.png")
    print("OK")

    step("Building imputer (segmenter='gradient_guided')...")
    from ImputerFactory import ImageImputerFactory
    factory = ImageImputerFactory()
    imputer = factory.build(
        model, processor, input_image, input_text,
        segmenter="gradient_guided",
    )
    print("OK")
    print(f"  layout = {imputer.layout}")

    step("Building game...")
    from Game.game_huggingface import VisionLanguageGame
    game = VisionLanguageGame(imputer, batch_size=64)
    n_img = game.n_players_image
    n_txt = game.n_players_text
    n_total = n_img + n_txt
    print(f"OK (n_total={n_total}, n_img={n_img}, n_txt={n_txt})")

    step("Value function test...")
    coalitions = np.zeros((2, n_total), dtype=bool)
    coalitions[1, :] = True
    result = game.value_function(coalitions=coalitions)
    print(f"OK (empty={result[0]:.4f}, full={result[1]:.4f})")

    step("forward_1d test...")
    coalitions_32 = np.random.RandomState(0).choice([True, False], size=(32, n_total))
    result_1d = imputer.forward_1d(coalitions_32, batch_size=16)
    print(f"OK (shape={result_1d.shape})")

    step("forward_crossmodal test...")
    coal_img = np.ones((4, n_img), dtype=bool)
    coal_txt = np.ones((4, n_txt), dtype=bool)
    result_cm = imputer.forward_crossmodal(coal_img, coal_txt)
    print(f"OK (shape={result_cm.shape})")

    step("Mask generation test...")
    seg = imputer.segmenter
    mask = seg.generate_masks(
        coalitions_image=np.ones((2, n_img), dtype=bool),
        coalitions_text=np.ones((2, n_txt), dtype=bool),
    )
    assert mask.image_binary_mask.shape == (2, 3, 224, 224), f"Bad image: {mask.image_binary_mask.shape}"
    assert mask.text_attention_mask.shape[0] == 2, f"Bad text: {mask.text_attention_mask.shape}"
    print("OK")

    step("Saliency storage test...")
    assert seg._saliency is not None, "_saliency not stored"
    assert seg._saliency.shape == (224, 224), f"Bad saliency shape: {seg._saliency.shape}"
    print("OK")

    print("\n>>> GradientGuidedSegmenter B2.1 test PASSED <<<")

except Exception:
    traceback.print_exc()
    sys.exit(1)
