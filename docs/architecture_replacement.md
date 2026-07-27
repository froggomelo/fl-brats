# Federated SAM architectural replacement

The fixed reference is `federated_unet_baseline`. Primary reports are U-Net
versus `federated_sam1`, and U-Net versus `federated_sam2`; SAM-to-SAM is only a
convenience row. SAM operates on complete axial 2D slice sets at evaluation,
then reconstructs WT, TC, and ET 3D volumes.

The primary adapter uses a federated learned 4-to-3 MRI projection, learned
region prompts, a frozen image encoder, and trainable prompt/projection/decoder
components. It never derives validation or test prompts from masks. The fixed
RGB alternative explicitly excludes T1 and is recorded in run metadata.

Pretrained weights are external, default to no automatic download, and must be
verified with `fedbrats verify-pretrained-model --model-config configs/model/sam1.yaml`.
`trainable_only` transmits only trainable state; reports must show both full
model and transmitted payload sizes.
