#!/bin/bash
set -e

echo "================================================="
echo " OurModel Face Detection - Docker Entrypoint"
echo "================================================="

case "$1" in
  evaluate)
    echo ""
    echo "===== Step 1: WiderFace Inference ====="
    echo "Model  : ./weights/mobilenet0.25_eca_cbam_Final.pth"
    echo "Network: mobile0.25"
    echo "Device : ${USE_CPU:+CPU}${USE_CPU:-GPU}"
    echo ""

    python tools/test_widerface.py \
      --trained_model ./weights/mobilenet0.25_eca_cbam_Final.pth \
      --network mobile0.25 \
      ${USE_CPU:+--cpu} \
      --dataset_folder ./data/widerface/val/images/ \
      --save_folder ./widerface_evaluate/widerface_txt/

    echo ""
    echo "===== Step 2: WiderFace AP Evaluation ====="
    cd widerface_evaluate
    python evaluation.py \
      -p ./widerface_txt/ \
      -g ./ground_truth/
    ;;

  detect)
    echo ""
    echo "===== Single Image Detection ====="
    echo "Model  : ./weights/mobilenet0.25_eca_cbam_Final.pth"
    echo "Network: mobile0.25"
    echo "Device : ${USE_CPU:+CPU}${USE_CPU:-GPU}"
    echo ""

    python tools/detect.py \
      --trained_model ./weights/mobilenet0.25_eca_cbam_Final.pth \
      --network mobile0.25 \
      ${USE_CPU:+--cpu}
    ;;

  shell)
    echo "Launching interactive shell..."
    exec /bin/bash
    ;;

  *)
    exec "$@"
    ;;
esac
