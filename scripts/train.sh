#!/bin/bash
# Train a model using EasyRec pipeline config
python -c "
from offline.training import OfflineTrainer
trainer = OfflineTrainer(pipeline_config_path='${1:-config/pipeline.config}')
result = trainer.train(model_dir='${2:-experiments/model}')
print(result)
"
