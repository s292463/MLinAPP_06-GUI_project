??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
~
Conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:N* 
shared_nameConv1D_1/kernel
w
#Conv1D_1/kernel/Read/ReadVariableOpReadVariableOpConv1D_1/kernel*"
_output_shapes
:N*
dtype0
r
Conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*
shared_nameConv1D_1/bias
k
!Conv1D_1/bias/Read/ReadVariableOpReadVariableOpConv1D_1/bias*
_output_shapes
:N*
dtype0
~
Conv1D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:N]* 
shared_nameConv1D_2/kernel
w
#Conv1D_2/kernel/Read/ReadVariableOpReadVariableOpConv1D_2/kernel*"
_output_shapes
:N]*
dtype0
r
Conv1D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*
shared_nameConv1D_2/bias
k
!Conv1D_2/bias/Read/ReadVariableOpReadVariableOpConv1D_2/bias*
_output_shapes
:]*
dtype0
z
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameDense_1/kernel
s
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel* 
_output_shapes
:
??*
dtype0
q
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_1/bias
j
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes	
:?*
dtype0
y
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?!*
shared_nameDense_2/kernel
r
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes
:	?!*
dtype0
p
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*
shared_nameDense_2/bias
i
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes
:!*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*'
shared_nameAdam/Conv1D_1/kernel/m
?
*Adam/Conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/m*"
_output_shapes
:N*
dtype0
?
Adam/Conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*%
shared_nameAdam/Conv1D_1/bias/m
y
(Adam/Conv1D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/m*
_output_shapes
:N*
dtype0
?
Adam/Conv1D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:N]*'
shared_nameAdam/Conv1D_2/kernel/m
?
*Adam/Conv1D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/m*"
_output_shapes
:N]*
dtype0
?
Adam/Conv1D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*%
shared_nameAdam/Conv1D_2/bias/m
y
(Adam/Conv1D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/m*
_output_shapes
:]*
dtype0
?
Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/Dense_1/kernel/m
?
)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/Dense_1/bias/m
x
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?!*&
shared_nameAdam/Dense_2/kernel/m
?
)Adam/Dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/m*
_output_shapes
:	?!*
dtype0
~
Adam/Dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*$
shared_nameAdam/Dense_2/bias/m
w
'Adam/Dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/m*
_output_shapes
:!*
dtype0
?
Adam/Conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*'
shared_nameAdam/Conv1D_1/kernel/v
?
*Adam/Conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/v*"
_output_shapes
:N*
dtype0
?
Adam/Conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*%
shared_nameAdam/Conv1D_1/bias/v
y
(Adam/Conv1D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/v*
_output_shapes
:N*
dtype0
?
Adam/Conv1D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:N]*'
shared_nameAdam/Conv1D_2/kernel/v
?
*Adam/Conv1D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/v*"
_output_shapes
:N]*
dtype0
?
Adam/Conv1D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:]*%
shared_nameAdam/Conv1D_2/bias/v
y
(Adam/Conv1D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/v*
_output_shapes
:]*
dtype0
?
Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/Dense_1/kernel/v
?
)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/Dense_1/bias/v
x
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?!*&
shared_nameAdam/Dense_2/kernel/v
?
)Adam/Dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/v*
_output_shapes
:	?!*
dtype0
~
Adam/Dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*$
shared_nameAdam/Dense_2/bias/v
w
'Adam/Dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/v*
_output_shapes
:!*
dtype0

NoOpNoOp
?9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?9
value?9B?9 B?9
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?
=iter

>beta_1

?beta_2
	@decay
Alearning_ratemm?m?m?-m?.m?7m?8m?v?v?v?v?-v?.v?7v?8v?
8
0
1
2
3
-4
.5
76
87
8
0
1
2
3
-4
.5
76
87
 
?
Bmetrics
Clayer_regularization_losses

Dlayers
	variables
trainable_variables
regularization_losses
Elayer_metrics
Fnon_trainable_variables
 
[Y
VARIABLE_VALUEConv1D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Gmetrics
Hlayer_regularization_losses

Ilayers
	variables
trainable_variables
regularization_losses
Jlayer_metrics
Knon_trainable_variables
 
 
 
?
Lmetrics
Mlayer_regularization_losses

Nlayers
	variables
trainable_variables
regularization_losses
Olayer_metrics
Pnon_trainable_variables
[Y
VARIABLE_VALUEConv1D_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Qmetrics
Rlayer_regularization_losses

Slayers
	variables
trainable_variables
regularization_losses
Tlayer_metrics
Unon_trainable_variables
 
 
 
?
Vmetrics
Wlayer_regularization_losses

Xlayers
!	variables
"trainable_variables
#regularization_losses
Ylayer_metrics
Znon_trainable_variables
 
 
 
?
[metrics
\layer_regularization_losses

]layers
%	variables
&trainable_variables
'regularization_losses
^layer_metrics
_non_trainable_variables
 
 
 
?
`metrics
alayer_regularization_losses

blayers
)	variables
*trainable_variables
+regularization_losses
clayer_metrics
dnon_trainable_variables
ZX
VARIABLE_VALUEDense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
emetrics
flayer_regularization_losses

glayers
/	variables
0trainable_variables
1regularization_losses
hlayer_metrics
inon_trainable_variables
 
 
 
?
jmetrics
klayer_regularization_losses

llayers
3	variables
4trainable_variables
5regularization_losses
mlayer_metrics
nnon_trainable_variables
ZX
VARIABLE_VALUEDense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
?
ometrics
player_regularization_losses

qlayers
9	variables
:trainable_variables
;regularization_losses
rlayer_metrics
snon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	vtotal
	wcount
x	variables
y	keras_api
D
	ztotal
	{count
|
_fn_kwargs
}	variables
~	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

v0
w1

x	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1

}	variables
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_18432
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv1D_1/kernel/Read/ReadVariableOp!Conv1D_1/bias/Read/ReadVariableOp#Conv1D_2/kernel/Read/ReadVariableOp!Conv1D_2/bias/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp"Dense_2/kernel/Read/ReadVariableOp Dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv1D_1/kernel/m/Read/ReadVariableOp(Adam/Conv1D_1/bias/m/Read/ReadVariableOp*Adam/Conv1D_2/kernel/m/Read/ReadVariableOp(Adam/Conv1D_2/bias/m/Read/ReadVariableOp)Adam/Dense_1/kernel/m/Read/ReadVariableOp'Adam/Dense_1/bias/m/Read/ReadVariableOp)Adam/Dense_2/kernel/m/Read/ReadVariableOp'Adam/Dense_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_1/kernel/v/Read/ReadVariableOp(Adam/Conv1D_1/bias/v/Read/ReadVariableOp*Adam/Conv1D_2/kernel/v/Read/ReadVariableOp(Adam/Conv1D_2/bias/v/Read/ReadVariableOp)Adam/Dense_1/kernel/v/Read/ReadVariableOp'Adam/Dense_1/bias/v/Read/ReadVariableOp)Adam/Dense_2/kernel/v/Read/ReadVariableOp'Adam/Dense_2/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_18873
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv1D_1/kernel/mAdam/Conv1D_1/bias/mAdam/Conv1D_2/kernel/mAdam/Conv1D_2/bias/mAdam/Dense_1/kernel/mAdam/Dense_1/bias/mAdam/Dense_2/kernel/mAdam/Dense_2/bias/mAdam/Conv1D_1/kernel/vAdam/Conv1D_1/bias/vAdam/Conv1D_2/kernel/vAdam/Conv1D_2/bias/vAdam/Dense_1/kernel/vAdam/Dense_1/bias/vAdam/Dense_2/kernel/vAdam/Dense_2/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_18982??
?

?
B__inference_Dense_2_layer_call_and_return_conditional_losses_18751

inputs1
matmul_readvariableop_resource:	?!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_18011

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_18662

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_180982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_Dense_2_layer_call_and_return_conditional_losses_18135

inputs1
matmul_readvariableop_resource:	?!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Z
?
@__inference_model_layer_call_and_return_conditional_losses_18596

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:N6
(conv1d_1_biasadd_readvariableop_resource:NJ
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:N]6
(conv1d_2_biasadd_readvariableop_resource:]:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?!5
'dense_2_biasadd_readvariableop_resource:!
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????N*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????N*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????N2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????N2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????N2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*/
_output_shapes
:?????????cN*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????cN*
squeeze_dims
2
MaxPooling1D_1/Squeeze?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsMaxPooling1D_1/Squeeze:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????cN2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N]*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N]2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^]*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*+
_output_shapes
:?????????^]*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:]*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^]2
Conv1D_2/BiasAddw
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^]2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^]2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????]*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????]*
squeeze_dims
2
MaxPooling1D_2/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????[  2
flatten/Const?
flatten/ReshapeReshapeMaxPooling1D_2/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ql??2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed2????2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 **i?>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulDense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapeDense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed220
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *<??>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?!*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
Dense_2/Softmax?
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_18091

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????[  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????]:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_18719

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_18731

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *<??>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_18122

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_18224

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ql??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 **i?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_18453

inputs
unknown:N
	unknown_0:N
	unknown_1:N]
	unknown_2:]
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?!
	unknown_6:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_181422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_18191

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *<??>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_18621

inputsA
+conv1d_expanddims_1_readvariableop_resource:N-
biasadd_readvariableop_resource:N
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????N*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????N*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:N*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????N2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????N2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_18657

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????[  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????]:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_18646

inputsA
+conv1d_expanddims_1_readvariableop_resource:N]-
biasadd_readvariableop_resource:]
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????cN2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N]*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N]2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^]*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????^]*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:]*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^]2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????^]2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????^]2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cN: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????cN
 
_user_specified_nameinputs
?&
?
@__inference_model_layer_call_and_return_conditional_losses_18142

inputs$
conv1d_1_18056:N
conv1d_1_18058:N$
conv1d_2_18079:N]
conv1d_2_18081:]!
dense_1_18112:
??
dense_1_18114:	? 
dense_2_18136:	?!
dense_2_18138:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_18056conv1d_1_18058*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_180552"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????cN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_180112 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_18079conv1d_2_18081*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^]*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_180782"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_180262 
MaxPooling1D_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_180912
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_180982
dropout/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_18112dense_1_18114*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_181112!
Dense_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_181222
dropout_1/PartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_18136dense_2_18138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_181352!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_2_layer_call_fn_18630

inputs
unknown:N]
	unknown_0:]
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^]*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_180782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????^]2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cN: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????cN
 
_user_specified_nameinputs
?
?
'__inference_Dense_1_layer_call_fn_18693

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_181112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_2_layer_call_fn_18032

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_180262
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?O
?
 __inference__wrapped_model_18002
input_1P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:N<
.model_conv1d_1_biasadd_readvariableop_resource:NP
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:N]<
.model_conv1d_2_biasadd_readvariableop_resource:]@
,model_dense_1_matmul_readvariableop_resource:
??<
-model_dense_1_biasadd_readvariableop_resource:	??
,model_dense_2_matmul_readvariableop_resource:	?!;
-model_dense_2_biasadd_readvariableop_resource:!
identity??%model/Conv1D_1/BiasAdd/ReadVariableOp?1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_2/BiasAdd/ReadVariableOp?1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?$model/Dense_1/BiasAdd/ReadVariableOp?#model/Dense_1/MatMul/ReadVariableOp?$model/Dense_2/BiasAdd/ReadVariableOp?#model/Dense_2/MatMul/ReadVariableOp?
$model/Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_1/conv1d/ExpandDims/dim?
 model/Conv1D_1/conv1d/ExpandDims
ExpandDimsinput_1-model/Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2"
 model/Conv1D_1/conv1d/ExpandDims?
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N*
dtype023
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_1/conv1d/ExpandDims_1/dim?
"model/Conv1D_1/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N2$
"model/Conv1D_1/conv1d/ExpandDims_1?
model/Conv1D_1/conv1dConv2D)model/Conv1D_1/conv1d/ExpandDims:output:0+model/Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????N*
paddingVALID*
strides
2
model/Conv1D_1/conv1d?
model/Conv1D_1/conv1d/SqueezeSqueezemodel/Conv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????N*
squeeze_dims

?????????2
model/Conv1D_1/conv1d/Squeeze?
%model/Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype02'
%model/Conv1D_1/BiasAdd/ReadVariableOp?
model/Conv1D_1/BiasAddBiasAdd&model/Conv1D_1/conv1d/Squeeze:output:0-model/Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????N2
model/Conv1D_1/BiasAdd?
model/Conv1D_1/ReluRelumodel/Conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????N2
model/Conv1D_1/Relu?
#model/MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_1/ExpandDims/dim?
model/MaxPooling1D_1/ExpandDims
ExpandDims!model/Conv1D_1/Relu:activations:0,model/MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????N2!
model/MaxPooling1D_1/ExpandDims?
model/MaxPooling1D_1/MaxPoolMaxPool(model/MaxPooling1D_1/ExpandDims:output:0*/
_output_shapes
:?????????cN*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_1/MaxPool?
model/MaxPooling1D_1/SqueezeSqueeze%model/MaxPooling1D_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????cN*
squeeze_dims
2
model/MaxPooling1D_1/Squeeze?
$model/Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_2/conv1d/ExpandDims/dim?
 model/Conv1D_2/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_1/Squeeze:output:0-model/Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????cN2"
 model/Conv1D_2/conv1d/ExpandDims?
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N]*
dtype023
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_2/conv1d/ExpandDims_1/dim?
"model/Conv1D_2/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N]2$
"model/Conv1D_2/conv1d/ExpandDims_1?
model/Conv1D_2/conv1dConv2D)model/Conv1D_2/conv1d/ExpandDims:output:0+model/Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^]*
paddingVALID*
strides
2
model/Conv1D_2/conv1d?
model/Conv1D_2/conv1d/SqueezeSqueezemodel/Conv1D_2/conv1d:output:0*
T0*+
_output_shapes
:?????????^]*
squeeze_dims

?????????2
model/Conv1D_2/conv1d/Squeeze?
%model/Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:]*
dtype02'
%model/Conv1D_2/BiasAdd/ReadVariableOp?
model/Conv1D_2/BiasAddBiasAdd&model/Conv1D_2/conv1d/Squeeze:output:0-model/Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^]2
model/Conv1D_2/BiasAdd?
model/Conv1D_2/ReluRelumodel/Conv1D_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^]2
model/Conv1D_2/Relu?
#model/MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_2/ExpandDims/dim?
model/MaxPooling1D_2/ExpandDims
ExpandDims!model/Conv1D_2/Relu:activations:0,model/MaxPooling1D_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^]2!
model/MaxPooling1D_2/ExpandDims?
model/MaxPooling1D_2/MaxPoolMaxPool(model/MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????]*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_2/MaxPool?
model/MaxPooling1D_2/SqueezeSqueeze%model/MaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????]*
squeeze_dims
2
model/MaxPooling1D_2/Squeeze{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????[  2
model/flatten/Const?
model/flatten/ReshapeReshape%model/MaxPooling1D_2/Squeeze:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshape?
model/dropout/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
model/dropout/Identity?
#model/Dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#model/Dense_1/MatMul/ReadVariableOp?
model/Dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Dense_1/MatMul?
$model/Dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/Dense_1/BiasAdd/ReadVariableOp?
model/Dense_1/BiasAddBiasAddmodel/Dense_1/MatMul:product:0,model/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/Dense_1/BiasAdd?
model/Dense_1/ReluRelumodel/Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/Dense_1/Relu?
model/dropout_1/IdentityIdentity model/Dense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_1/Identity?
#model/Dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?!*
dtype02%
#model/Dense_2/MatMul/ReadVariableOp?
model/Dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/MatMul?
$model/Dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02&
$model/Dense_2/BiasAdd/ReadVariableOp?
model/Dense_2/BiasAddBiasAddmodel/Dense_2/MatMul:product:0,model/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/BiasAdd?
model/Dense_2/SoftmaxSoftmaxmodel/Dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/Softmax?
IdentityIdentitymodel/Dense_2/Softmax:softmax:0&^model/Conv1D_1/BiasAdd/ReadVariableOp2^model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_2/BiasAdd/ReadVariableOp2^model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp%^model/Dense_1/BiasAdd/ReadVariableOp$^model/Dense_1/MatMul/ReadVariableOp%^model/Dense_2/BiasAdd/ReadVariableOp$^model/Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2N
%model/Conv1D_1/BiasAdd/ReadVariableOp%model/Conv1D_1/BiasAdd/ReadVariableOp2f
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_2/BiasAdd/ReadVariableOp%model/Conv1D_2/BiasAdd/ReadVariableOp2f
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2L
$model/Dense_1/BiasAdd/ReadVariableOp$model/Dense_1/BiasAdd/ReadVariableOp2J
#model/Dense_1/MatMul/ReadVariableOp#model/Dense_1/MatMul/ReadVariableOp2L
$model/Dense_2/BiasAdd/ReadVariableOp$model/Dense_2/BiasAdd/ReadVariableOp2J
#model/Dense_2/MatMul/ReadVariableOp#model/Dense_2/MatMul/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
%__inference_model_layer_call_fn_18161
input_1
unknown:N
	unknown_0:N
	unknown_1:N]
	unknown_2:]
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?!
	unknown_6:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_181422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
%__inference_model_layer_call_fn_18474

inputs
unknown:N
	unknown_0:N
	unknown_1:N]
	unknown_2:]
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?!
	unknown_6:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_183052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_18078

inputsA
+conv1d_expanddims_1_readvariableop_resource:N]-
biasadd_readvariableop_resource:]
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????cN2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N]*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N]2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^]*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????^]*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:]*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^]2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????^]2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????^]2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cN: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????cN
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_18098

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_18672

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_18026

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_1_layer_call_fn_18605

inputs
unknown:N
	unknown_0:N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_180552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_18345
input_1
unknown:N
	unknown_0:N
	unknown_1:N]
	unknown_2:]
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?!
	unknown_6:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_183052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
E
)__inference_dropout_1_layer_call_fn_18709

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_181222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_18055

inputsA
+conv1d_expanddims_1_readvariableop_resource:N-
biasadd_readvariableop_resource:N
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????N*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????N*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:N*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????N2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????N2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?G
?
@__inference_model_layer_call_and_return_conditional_losses_18528

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:N6
(conv1d_1_biasadd_readvariableop_resource:NJ
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:N]6
(conv1d_2_biasadd_readvariableop_resource:]:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?!5
'dense_2_biasadd_readvariableop_resource:!
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????N*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????N*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????N2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????N2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????N2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*/
_output_shapes
:?????????cN*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????cN*
squeeze_dims
2
MaxPooling1D_1/Squeeze?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsMaxPooling1D_1/Squeeze:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????cN2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:N]*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:N]2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????^]*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*+
_output_shapes
:?????????^]*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:]*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????^]2
Conv1D_2/BiasAddw
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^]2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????^]2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????]*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????]*
squeeze_dims
2
MaxPooling1D_2/Squeezeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????[  2
flatten/Const?
flatten/ReshapeReshapeMaxPooling1D_2/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMuldropout/Identity:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_1/Relu?
dropout_1/IdentityIdentityDense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_1/Identity?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?!*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMuldropout_1/Identity:output:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
Dense_2/Softmax?
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_18667

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_182242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_18684

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ql??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 **i?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Dense_2_layer_call_fn_18740

inputs
unknown:	?!
	unknown_0:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_181352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_18714

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_181912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
@__inference_model_layer_call_and_return_conditional_losses_18374
input_1$
conv1d_1_18348:N
conv1d_1_18350:N$
conv1d_2_18354:N]
conv1d_2_18356:]!
dense_1_18362:
??
dense_1_18364:	? 
dense_2_18368:	?!
dense_2_18370:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_18348conv1d_1_18350*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_180552"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????cN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_180112 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_18354conv1d_2_18356*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^]*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_180782"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_180262 
MaxPooling1D_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_180912
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_180982
dropout/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_18362dense_1_18364*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_181112!
Dense_1/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_181222
dropout_1/PartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_18368dense_2_18370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_181352!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_18111

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?G
?
__inference__traced_save_18873
file_prefix.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :N:N:N]:]:
??:?:	?!:!: : : : : : : : : :N:N:N]:]:
??:?:	?!:!:N:N:N]:]:
??:?:	?!:!: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:N: 

_output_shapes
:N:($
"
_output_shapes
:N]: 

_output_shapes
:]:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?!: 

_output_shapes
:!:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:N: 

_output_shapes
:N:($
"
_output_shapes
:N]: 

_output_shapes
:]:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?!: 

_output_shapes
:!:($
"
_output_shapes
:N: 

_output_shapes
:N:($
"
_output_shapes
:N]: 

_output_shapes
:]:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?!: !

_output_shapes
:!:"

_output_shapes
: 
?
J
.__inference_MaxPooling1D_1_layer_call_fn_18017

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_180112
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_18982
file_prefix6
 assignvariableop_conv1d_1_kernel:N.
 assignvariableop_1_conv1d_1_bias:N8
"assignvariableop_2_conv1d_2_kernel:N].
 assignvariableop_3_conv1d_2_bias:]5
!assignvariableop_4_dense_1_kernel:
??.
assignvariableop_5_dense_1_bias:	?4
!assignvariableop_6_dense_2_kernel:	?!-
assignvariableop_7_dense_2_bias:!&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: @
*assignvariableop_17_adam_conv1d_1_kernel_m:N6
(assignvariableop_18_adam_conv1d_1_bias_m:N@
*assignvariableop_19_adam_conv1d_2_kernel_m:N]6
(assignvariableop_20_adam_conv1d_2_bias_m:]=
)assignvariableop_21_adam_dense_1_kernel_m:
??6
'assignvariableop_22_adam_dense_1_bias_m:	?<
)assignvariableop_23_adam_dense_2_kernel_m:	?!5
'assignvariableop_24_adam_dense_2_bias_m:!@
*assignvariableop_25_adam_conv1d_1_kernel_v:N6
(assignvariableop_26_adam_conv1d_1_bias_v:N@
*assignvariableop_27_adam_conv1d_2_kernel_v:N]6
(assignvariableop_28_adam_conv1d_2_bias_v:]=
)assignvariableop_29_adam_dense_1_kernel_v:
??6
'assignvariableop_30_adam_dense_1_bias_v:	?<
)assignvariableop_31_adam_dense_2_kernel_v:	?!5
'assignvariableop_32_adam_dense_2_bias_v:!
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv1d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv1d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv1d_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv1d_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
#__inference_signature_wrapper_18432
input_1
unknown:N
	unknown_0:N
	unknown_1:N]
	unknown_2:]
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?!
	unknown_6:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_180022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?)
?
@__inference_model_layer_call_and_return_conditional_losses_18305

inputs$
conv1d_1_18279:N
conv1d_1_18281:N$
conv1d_2_18285:N]
conv1d_2_18287:]!
dense_1_18293:
??
dense_1_18295:	? 
dense_2_18299:	?!
dense_2_18301:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_18279conv1d_1_18281*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_180552"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????cN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_180112 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_18285conv1d_2_18287*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^]*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_180782"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_180262 
MaxPooling1D_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_180912
flatten/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_182242!
dropout/StatefulPartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_18293dense_1_18295*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_181112!
Dense_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_181912#
!dropout_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_18299dense_2_18301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_181352!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
@__inference_model_layer_call_and_return_conditional_losses_18403
input_1$
conv1d_1_18377:N
conv1d_1_18379:N$
conv1d_2_18383:N]
conv1d_2_18385:]!
dense_1_18391:
??
dense_1_18393:	? 
dense_2_18397:	?!
dense_2_18399:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_18377conv1d_1_18379*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_180552"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????cN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_180112 
MaxPooling1D_1/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_18383conv1d_2_18385*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????^]*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_180782"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????]* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_180262 
MaxPooling1D_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_180912
flatten/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_182242!
dropout/StatefulPartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_18391dense_1_18393*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_181112!
Dense_1/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_181912#
!dropout_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_18397dense_2_18399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_181352!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_18704

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_18651

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_180912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????]:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????;
Dense_20
StatefulPartitionedCall:0?????????!tensorflow/serving/predict:??
?R
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 78, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 93, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3738492116191333, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 170, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.36848627854834504, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}, "shared_object_id": 18, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 78, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 93, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3738492116191333, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 170, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.36848627854834504, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 17}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 20}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001290000043809414, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 78, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 22}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 93, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 78}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 99, 78]}}
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 24}}
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 25}}
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3738492116191333, "noise_shape": null, "seed": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 10}
?	

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 170, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2139}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2139]}}
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.36848627854834504, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 14}
?	

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "Dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 170}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 170]}}
?
=iter

>beta_1

?beta_2
	@decay
Alearning_ratemm?m?m?-m?.m?7m?8m?v?v?v?v?-v?.v?7v?8v?"
	optimizer
X
0
1
2
3
-4
.5
76
87"
trackable_list_wrapper
X
0
1
2
3
-4
.5
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bmetrics
Clayer_regularization_losses

Dlayers
	variables
trainable_variables
regularization_losses
Elayer_metrics
Fnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#N2Conv1D_1/kernel
:N2Conv1D_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gmetrics
Hlayer_regularization_losses

Ilayers
	variables
trainable_variables
regularization_losses
Jlayer_metrics
Knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lmetrics
Mlayer_regularization_losses

Nlayers
	variables
trainable_variables
regularization_losses
Olayer_metrics
Pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#N]2Conv1D_2/kernel
:]2Conv1D_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qmetrics
Rlayer_regularization_losses

Slayers
	variables
trainable_variables
regularization_losses
Tlayer_metrics
Unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vmetrics
Wlayer_regularization_losses

Xlayers
!	variables
"trainable_variables
#regularization_losses
Ylayer_metrics
Znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[metrics
\layer_regularization_losses

]layers
%	variables
&trainable_variables
'regularization_losses
^layer_metrics
_non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
`metrics
alayer_regularization_losses

blayers
)	variables
*trainable_variables
+regularization_losses
clayer_metrics
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2Dense_1/kernel
:?2Dense_1/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
emetrics
flayer_regularization_losses

glayers
/	variables
0trainable_variables
1regularization_losses
hlayer_metrics
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
jmetrics
klayer_regularization_losses

llayers
3	variables
4trainable_variables
5regularization_losses
mlayer_metrics
nnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?!2Dense_2/kernel
:!2Dense_2/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ometrics
player_regularization_losses

qlayers
9	variables
:trainable_variables
;regularization_losses
rlayer_metrics
snon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	vtotal
	wcount
x	variables
y	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 28}
?
	ztotal
	{count
|
_fn_kwargs
}	variables
~	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 20}
:  (2total
:  (2count
.
v0
w1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
*:(N2Adam/Conv1D_1/kernel/m
 :N2Adam/Conv1D_1/bias/m
*:(N]2Adam/Conv1D_2/kernel/m
 :]2Adam/Conv1D_2/bias/m
':%
??2Adam/Dense_1/kernel/m
 :?2Adam/Dense_1/bias/m
&:$	?!2Adam/Dense_2/kernel/m
:!2Adam/Dense_2/bias/m
*:(N2Adam/Conv1D_1/kernel/v
 :N2Adam/Conv1D_1/bias/v
*:(N]2Adam/Conv1D_2/kernel/v
 :]2Adam/Conv1D_2/bias/v
':%
??2Adam/Dense_1/kernel/v
 :?2Adam/Dense_1/bias/v
&:$	?!2Adam/Dense_2/kernel/v
:!2Adam/Dense_2/bias/v
?2?
 __inference__wrapped_model_18002?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
%__inference_model_layer_call_fn_18161
%__inference_model_layer_call_fn_18453
%__inference_model_layer_call_fn_18474
%__inference_model_layer_call_fn_18345?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_18528
@__inference_model_layer_call_and_return_conditional_losses_18596
@__inference_model_layer_call_and_return_conditional_losses_18374
@__inference_model_layer_call_and_return_conditional_losses_18403?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Conv1D_1_layer_call_fn_18605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_18621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_1_layer_call_fn_18017?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_18011?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
(__inference_Conv1D_2_layer_call_fn_18630?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_18646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_2_layer_call_fn_18032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_18026?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
'__inference_flatten_layer_call_fn_18651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_18657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_18662
'__inference_dropout_layer_call_fn_18667?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_18672
B__inference_dropout_layer_call_and_return_conditional_losses_18684?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_Dense_1_layer_call_fn_18693?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_1_layer_call_and_return_conditional_losses_18704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_18709
)__inference_dropout_1_layer_call_fn_18714?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_18719
D__inference_dropout_1_layer_call_and_return_conditional_losses_18731?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_Dense_2_layer_call_fn_18740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_2_layer_call_and_return_conditional_losses_18751?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_18432input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_18621f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????N
? ?
(__inference_Conv1D_1_layer_call_fn_18605Y4?1
*?'
%?"
inputs??????????
? "???????????N?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_18646d3?0
)?&
$?!
inputs?????????cN
? ")?&
?
0?????????^]
? ?
(__inference_Conv1D_2_layer_call_fn_18630W3?0
)?&
$?!
inputs?????????cN
? "??????????^]?
B__inference_Dense_1_layer_call_and_return_conditional_losses_18704^-.0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_Dense_1_layer_call_fn_18693Q-.0?-
&?#
!?
inputs??????????
? "????????????
B__inference_Dense_2_layer_call_and_return_conditional_losses_18751]780?-
&?#
!?
inputs??????????
? "%?"
?
0?????????!
? {
'__inference_Dense_2_layer_call_fn_18740P780?-
&?#
!?
inputs??????????
? "??????????!?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_18011?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_1_layer_call_fn_18017wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_18026?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_2_layer_call_fn_18032wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
 __inference__wrapped_model_18002t-.785?2
+?(
&?#
input_1??????????
? "1?.
,
Dense_2!?
Dense_2?????????!?
D__inference_dropout_1_layer_call_and_return_conditional_losses_18719^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_18731^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ~
)__inference_dropout_1_layer_call_fn_18709Q4?1
*?'
!?
inputs??????????
p 
? "???????????~
)__inference_dropout_1_layer_call_fn_18714Q4?1
*?'
!?
inputs??????????
p
? "????????????
B__inference_dropout_layer_call_and_return_conditional_losses_18672^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_18684^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_18662Q4?1
*?'
!?
inputs??????????
p 
? "???????????|
'__inference_dropout_layer_call_fn_18667Q4?1
*?'
!?
inputs??????????
p
? "????????????
B__inference_flatten_layer_call_and_return_conditional_losses_18657]3?0
)?&
$?!
inputs?????????]
? "&?#
?
0??????????
? {
'__inference_flatten_layer_call_fn_18651P3?0
)?&
$?!
inputs?????????]
? "????????????
@__inference_model_layer_call_and_return_conditional_losses_18374p-.78=?:
3?0
&?#
input_1??????????
p 

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_18403p-.78=?:
3?0
&?#
input_1??????????
p

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_18528o-.78<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_18596o-.78<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????!
? ?
%__inference_model_layer_call_fn_18161c-.78=?:
3?0
&?#
input_1??????????
p 

 
? "??????????!?
%__inference_model_layer_call_fn_18345c-.78=?:
3?0
&?#
input_1??????????
p

 
? "??????????!?
%__inference_model_layer_call_fn_18453b-.78<?9
2?/
%?"
inputs??????????
p 

 
? "??????????!?
%__inference_model_layer_call_fn_18474b-.78<?9
2?/
%?"
inputs??????????
p

 
? "??????????!?
#__inference_signature_wrapper_18432-.78@?=
? 
6?3
1
input_1&?#
input_1??????????"1?.
,
Dense_2!?
Dense_2?????????!