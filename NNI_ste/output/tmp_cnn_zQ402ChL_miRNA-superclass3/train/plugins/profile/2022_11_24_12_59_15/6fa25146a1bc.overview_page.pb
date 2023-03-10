�	Dl�p��@Dl�p��@!Dl�p��@	*�5�@*�5�@!*�5�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLDl�p��@��ۂ���?1'ݖ�@A�oC�׼�?It���@YIV�F�?rEagerKernelExecute 0*	���Mb�e@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����Li�?!ߡ�?cx@@)X)�k{�?1I�T��>@:Preprocessing2F
Iterator::Modela7l[�ٴ?!oW>oZG@)dyW=`�?1f\Q{�@=@:Preprocessing2U
Iterator::Model::ParallelMapV2���)�?!wR+cjs1@)���)�?1wR+cjs1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��f�R@�?!�K��f-@)�y���?1��uH&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��l˷?!�����J@)-@�j�?1
�W��j@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��Bs�Fz?!�6��m@)��Bs�Fz?1�6��m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��J̳��?!x]���A@)P�mp�b?13s��l��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor� !��`?!+�v3��?)� !��`?1+�v3��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceZ.��S\?!,Lk�%��?)Z.��S\?1,Lk�%��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�45.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9*�5�@Iz��
P@QF��]^@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ۂ���?��ۂ���?!��ۂ���?      ��!       "	'ݖ�@'ݖ�@!'ݖ�@*      ��!       2	�oC�׼�?�oC�׼�?!�oC�׼�?:	t���@t���@!t���@B      ��!       J	IV�F�?IV�F�?!IV�F�?R      ��!       Z	IV�F�?IV�F�?!IV�F�?b      ��!       JGPUY*�5�@b qz��
P@yF��]^@@�"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�,K��?!�,K��?0"1
model/Conv1D_2/conv1dConv2D�fh�L��?!���;���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterg����?!�~m�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�7���?!�������?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradX�2��Ӣ?!���I�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad.*�,}-�?!0���;�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputC�q�ǝ?!��WT�?0"1
model/Conv1D_3/conv1dConv2D�:w7튝?!��u+��?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad/�8VG��?!�٠���?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�1�W�?! ^�f�?0Q      Y@Y9��8�c*@a9��8��U@qRs.�}C@y�@m�`�?"�
both�Your program is POTENTIALLY input-bound because 18.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�45.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�39.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 