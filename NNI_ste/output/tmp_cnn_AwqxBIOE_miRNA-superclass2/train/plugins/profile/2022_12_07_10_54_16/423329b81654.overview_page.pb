�	J��P�B @J��P�B @!J��P�B @	�mIʇa@�mIʇa@!�mIʇa@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCJ��P�B @Z�{,��?1�)ʥ�@I��=��@Y辜ٮ��?rEagerKernelExecute 0*	��Mb�d@2F
Iterator::ModelTT�J�?!u_;�KG@) <�Bus�?1��fŭ�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����'�?!�]�1~'<@)` �C���?1�1��8@:Preprocessing2U
Iterator::Model::ParallelMapV2V�pA��?!0X�\�0@)V�pA��?10X�\�0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip |(Ѷ?!����z�J@)����?1�*�E�3%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicezq�ŉ?!��N@@)@)zq�ŉ?1��N@@)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�!�{��?!����Y*@)c��K�A�?1XFJ��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor �t���{?!���x;@) �t���{?1���x;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_�D�
�?!�E�iO-@)�SH�9d?1��ŝ*��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�27.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�mIʇa@IQ��ާ�A@Qխ��?�N@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Z�{,��?Z�{,��?!Z�{,��?      ��!       "	�)ʥ�@�)ʥ�@!�)ʥ�@*      ��!       2      ��!       :	��=��@��=��@!��=��@B      ��!       J	辜ٮ��?辜ٮ��?!辜ٮ��?R      ��!       Z	辜ٮ��?辜ٮ��?!辜ٮ��?b      ��!       JGPUY�mIʇa@b qQ��ާ�A@yխ��?�N@�"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad`��T(�?!`��T(�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad{�y!�#�?!Ra���?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���*}��?!��h=�j�?0"1
model/Conv1D_1/conv1dConv2Dc���Uק?!�Hٹ�`�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�w"�Kȧ?!?��h��?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposepe>�ŧ?!���!��?"3
model/Conv1D_1/BiasAddBiasAdd�K���9�?!fIz`��?"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose1Rd+�Φ?!�ӊ;c�?"-
model/Conv1D_1/ReluRelu�:�K��?!�,R���?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transposeq�0���?!|G�J=��?Q      Y@Y!Y�B*@a����7�U@q��vq5� @yږg�,�?"�

both�Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�27.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 