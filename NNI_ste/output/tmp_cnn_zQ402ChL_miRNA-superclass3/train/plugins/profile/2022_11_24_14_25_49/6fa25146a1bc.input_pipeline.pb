	�]�o� @�]�o� @!�]�o� @      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�]�o� @�uS�ke�?1�^(`;8@A>]ݱ�&�?I���|@@@rEagerKernelExecute 0*	��S㥎�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�<�+J	�?!��9bg�E@)5ӽN���?1u2_�C@:Preprocessing2F
Iterator::Model }��A��?!��D�>@)�w��x[�?1�Uz(�2;@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map��h���?!����j5@)^gE�D�?1�+�@*/@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatW�'��?!R��~@)�BX�%��?1���͋@:Preprocessing2U
Iterator::Model::ParallelMapV2�)��F��?!�Ř��@)�)��F��?1�Ř��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN�����?!ЋO��@)�cϞ˔?1T��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��U��?!P���G@)��PN���?1~�0dӒ @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;��Tގ�?!��s(v @)5&�\R�?1!'���2�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�&S��?!{I5��?)�&S��?1{I5��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ʅʿ�w?!!��ڋs�?)�ʅʿ�w?1!��ڋs�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range :̗`o?!����D1�?) :̗`o?1����D1�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�,{�l?!������?)�,{�l?1������?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate&U�M�Ms?!^.�1!1�?)��A��S?1(0!wj�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensory�&1�L?!�o�H>��?)y�&1�L?1�o�H>��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIkx���hQ@QU%�]>@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�uS�ke�?�uS�ke�?!�uS�ke�?      ��!       "	�^(`;8@�^(`;8@!�^(`;8@*      ��!       2	>]ݱ�&�?>]ݱ�&�?!>]ݱ�&�?:	���|@@@���|@@@!���|@@@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qkx���hQ@yU%�]>@