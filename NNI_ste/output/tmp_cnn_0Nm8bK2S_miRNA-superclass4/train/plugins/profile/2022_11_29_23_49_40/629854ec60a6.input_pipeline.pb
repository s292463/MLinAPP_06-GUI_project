	??yT??@??yT??@!??yT??@	D???
@D???
@!D???
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??yT??@??????1????o[@I????C@Y$?P29???rEagerKernelExecute 0*	??n??c@2F
Iterator::Model?W?????!???F	G@)?im?k??1wB?v???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!?@?;??@@)?x?0DN??1?M?vߥ<@:Preprocessing2U
Iterator::Model::ParallelMapV2*??.???!}o-??-@)*??.???1}o-??-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?-X???!?O?į?@)?-X???1?O?į?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipE???????!?Bpy??J@)T1??c??14rZ%P@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????! Ҫ?#v)@)?z?ۡa??1XT?R?]@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA	]?|?!??X4?@)A	]?|?1??X4?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????
???!?????'-@)иp $h?1]???
???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?37.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9D???
@I??Uܼ?C@Q9??nL@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????!??????      ??!       "	????o[@????o[@!????o[@*      ??!       2      ??!       :	????C@????C@!????C@B      ??!       J	$?P29???$?P29???!$?P29???R      ??!       Z	$?P29???$?P29???!$?P29???b      ??!       JGPUYD???
@b q??Uܼ?C@y9??nL@