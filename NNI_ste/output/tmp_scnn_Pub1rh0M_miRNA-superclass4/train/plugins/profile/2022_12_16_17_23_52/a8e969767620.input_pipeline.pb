	G;n???O@G;n???O@!G;n???O@	?c?w @?c?w @!?c?w @"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G;n???O@?
G?J???1?????!F@I?&???%1@Y ???Q#??r0*	LbX9,c@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?z?p̲??!?'1L->@){k`????1?A~[??8@:Preprocessing2U
Iterator::Model::ParallelMapV2*????1??!ܧ?Z@+7@)*????1??1ܧ?Z@+7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapӟ?H??!??1??#<@)?:??????1Qu?fa/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice3?`???!E?r??(@)3?`???1E?r??(@:Preprocessing2F
Iterator::Model??Y.???!??͔A@)?D??f֒?1cg ??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Z}u??!}3@?5P@)?%??s|??1??+%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory=????!??V[*@)y=????1??V[*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?26.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?c?w @I???#B<@Q??C{kNQ@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?
G?J????
G?J???!?
G?J???      ??!       "	?????!F@?????!F@!?????!F@*      ??!       2      ??!       :	?&???%1@?&???%1@!?&???%1@B      ??!       J	 ???Q#?? ???Q#??! ???Q#??R      ??!       Z	 ???Q#?? ???Q#??! ???Q#??b      ??!       JGPUY?c?w @b q???#B<@y??C{kNQ@