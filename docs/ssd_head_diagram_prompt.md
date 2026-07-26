Draw two diagrams side by side comparing SSD extra feature block architectures.

**Before (our old design):**
Block takes input (e.g., 1280 channels) and applies:
1. Depthwise Conv 3×3, stride=2 (operates on all 1280 channels) + BN + ReLU6
2. Pointwise Conv 1×1 → 512 channels + BN + ReLU6

**After (reference inverted bottleneck):**
Block takes input (e.g., 1280 channels) and applies:
1. Pointwise Conv 1×1 → 256 channels (squeeze to half of output) + BN + ReLU6
2. Depthwise Conv 3×3, stride=2 (operates on only 256 channels) + BN + ReLU6
3. Pointwise Conv 1×1 → 512 channels (expand) + BN + ReLU6

Show the channel dimensions at each step. There are 4 such blocks chained: 1280→512→256→256→128.

Also show the full SSD head architecture:

**Backbone (MobileNetV2):**
- Input: 320×320×3
- Tap 1: Layer 14 expansion → 20×20×576
- Tap 2: Final output → 10×10×1280

**Extra blocks (4 blocks, show before/after designs):**
- Block 0: 10×10×1280 → 5×5×512
- Block 1: 5×5×512 → 3×3×256
- Block 2: 3×3×256 → 2×2×256
- Block 3: 2×2×256 → 1×1×128

**Prediction heads (applied to each of the 6 feature maps):**
Each feature map gets two parallel depthwise separable conv branches:
- Localization: DW 3×3 + BN + ReLU6 → PW 1×1 → K×4 outputs (K anchors × 4 box offsets)
- Classification: DW 3×3 + BN + ReLU6 → PW 1×1 → K×81 outputs (K anchors × 81 classes)

Anchors per level:
- Level 0 (20×20×576): 4 anchors/cell
- Levels 1-5: 6 anchors/cell each
- Total: 4×400 + 6×(100+25+9+4+1) = 2434 anchors
