# Dataset placement

The supplied Roboflow export is stored in this folder with its existing split. Expected layout:

```text
dataset/raw/
├── train/{Cotton,Cotton MIxed,Denim,Polyester,Silk,Viscose,Wool}/
├── valid/{Cotton,Cotton MIxed,Denim,Polyester,Silk,Viscose,Wool}/
└── test/{Cotton,Cotton MIxed,Denim,Polyester,Silk,Viscose,Wool}/
```

The pipeline normalizes `Cotton MIxed` to the canonical class name `Cotton Mixed`. It preserves these supplied splits. Keep the dataset out of Git. If downloading a different archive, inspect its structure and source metadata before replacing this export.

