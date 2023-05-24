from ptq2theMoon.ptqTransformerBased import PTQForTransformerBased

# The calibration data path
calib = r'E:\semester2\5005B\code\data\ImageNet12_1k\ILSVRC2012_img_calib'
# The validation data path
val = r'E:\semester2\5005B\code\data\ImageNet12_1k\ILSVRC2012_img_val'
# Your dir path to store profiler result that can be visualized by tensorboard
profiler_path = 'logs'
ptq_tool = PTQForTransformerBased(calibDir=calib)

# model name, choose in ['vit_base_patch16_224', 'vit_large_patch16_224']
model_name = 'vit_base_patch16_224'

if __name__ == '__main__':
    # post training quantization
    quantized_graph = ptq_tool.ptq(batchsize=2, model_name=model_name)
    # show profiler table of different operation
    # ptq_tool.profiler_table(out_path=profiler_path, model_name=model_name, graph=None)
    # ptq_tool.profiler_table(out_path=profiler_path, graph=quantized_graph, model_name=None)

    # layerwise report by ppq
    report = ptq_tool.report('layerwise', quantized_graph)

    # perform evaluation
    eval_report = ptq_tool.evaluation(val_dir=val, graph=quantized_graph, batchsize=2)
