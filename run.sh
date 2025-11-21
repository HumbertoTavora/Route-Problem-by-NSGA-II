POP_SIZE=$1
CROSS_PROB=$2
MUT_PROB=$3
NUM_GEN=$4

PREFIX="Input_Data_pop${POP_SIZE}_crossProb${CROSS_PROB}_mutProb${MUT_PROB}_numGen${NUM_GEN}"
RESULTS_FILE="results/${PREFIX}.csv"
VIZ_DIR="visualization/${PREFIX}"
OUTPUT_DIR_FIGURES="visualization/${PREFIX}/figures"
FRAME_DIR_EVOLUTION="visualization/${PREFIX}/frames/evolution"
FRAME_DIR_ROUTE="visualization/${PREFIX}/frames/route"
OUTPUT_DIR_GIF="visualization/${PREFIX}/gifs"




python runAlgorithm.py \
  --popSize=$POP_SIZE \
  --crossProb=$CROSS_PROB \
  --mutProb=$MUT_PROB \
  --numGen=$NUM_GEN

python create_figures.py \
  --results_file "$RESULTS_FILE" \
  --output_dir "$VIZ_DIR/figures"


python create_evolution_gif.py \
  --results_file "$RESULTS_FILE" \
  --frames_dir "$FRAME_DIR_EVOLUTION" \
  --output_dir "$OUTPUT_DIR_GIF" \
  --fps 5

python create_route_gif.py \
  --instance_json "data/json/Input_Data.json" \
  --results_file "$RESULTS_FILE" \
  --frames_dir "$FRAME_DIR_ROUTE" \
  --output_dir "$OUTPUT_DIR_GIF" \
  --fps 2
