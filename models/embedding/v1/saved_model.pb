еп
§р
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЭЬL>"
Ttype0:
2
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
7
Square
x"T
y"T"
Ttype:
2	
С
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
executor_typestring Ј
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.02unknown8ЂП
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:
*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*$
shared_nameoutput_layer/kernel
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	
*
dtype0
{
hidden_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namehidden_layer/bias
t
%hidden_layer/bias/Read/ReadVariableOpReadVariableOphidden_layer/bias*
_output_shapes	
:*
dtype0

hidden_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namehidden_layer/kernel
}
'hidden_layer/kernel/Read/ReadVariableOpReadVariableOphidden_layer/kernel* 
_output_shapes
:
*
dtype0

serving_default_flatten_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_inputhidden_layer/kernelhidden_layer/biasoutput_layer/kerneloutput_layer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_6221642

NoOpNoOp
 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Л
valueБBЎ BЇ
В
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
І
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias*

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0axis* 
 
0
1
(2
)3*
 
0
1
(2
)3*

10
21* 
А
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
8trace_0
9trace_1
:trace_2
;trace_3* 
6
<trace_0
=trace_1
>trace_2
?trace_3* 
* 

@serving_default* 
* 
* 
* 

Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ftrace_0* 

Gtrace_0* 

0
1*

0
1*
	
10* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
c]
VARIABLE_VALUEhidden_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhidden_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

Ttrace_0* 

Utrace_0* 

(0
)1*

(0
)1*
	
20* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

btrace_0* 

ctrace_0* 
* 

dtrace_0* 

etrace_0* 
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
10* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
20* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehidden_layer/kernelhidden_layer/biasoutput_layer/kerneloutput_layer/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_6221892
ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer/kernelhidden_layer/biasoutput_layer/kerneloutput_layer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_6221914Ч
ц
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221409

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:џџџџџџџџџ*
alpha%
з#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

г
%__inference_signature_wrapper_6221642
flatten_input
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_6221372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameflatten_input
Я

.__inference_output_layer_layer_call_fn_6221797

inputs
unknown:	

	unknown_0:

identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6221425o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А	
k
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221442

inputs
identityW
l2_normalize/SquareSquareinputs*
T0*'
_output_shapes
:џџџџџџџџџ
l
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџg
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџe
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentityl2_normalize:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
,
ц
G__inference_sequential_layer_call_and_return_conditional_losses_6221744

inputs?
+hidden_layer_matmul_readvariableop_resource:
;
,hidden_layer_biasadd_readvariableop_resource:	>
+output_layer_matmul_readvariableop_resource:	
:
,output_layer_biasadd_readvariableop_resource:

identityЂ#hidden_layer/BiasAdd/ReadVariableOpЂ"hidden_layer/MatMul/ReadVariableOpЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
"hidden_layer/MatMul/ReadVariableOpReadVariableOp+hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
hidden_layer/MatMulMatMulflatten/Reshape:output:0*hidden_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
#hidden_layer/BiasAdd/ReadVariableOpReadVariableOp,hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
hidden_layer/BiasAddBiasAddhidden_layer/MatMul:product:0+hidden_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ{
leaky_re_lu/LeakyRelu	LeakyReluhidden_layer/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
alpha%
з#<
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0 
output_layer/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

&unit_normalization/l2_normalize/SquareSquareoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

5unit_normalization/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:й
#unit_normalization/l2_normalize/SumSum*unit_normalization/l2_normalize/Square:y:0>unit_normalization/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(n
)unit_normalization/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+Ц
'unit_normalization/l2_normalize/MaximumMaximum,unit_normalization/l2_normalize/Sum:output:02unit_normalization/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%unit_normalization/l2_normalize/RsqrtRsqrt+unit_normalization/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
unit_normalization/l2_normalizeMuloutput_layer/BiasAdd:output:0)unit_normalization/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ђ
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#unit_normalization/l2_normalize:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
NoOpNoOp$^hidden_layer/BiasAdd/ReadVariableOp#^hidden_layer/MatMul/ReadVariableOp6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2J
#hidden_layer/BiasAdd/ReadVariableOp#hidden_layer/BiasAdd/ReadVariableOp2H
"hidden_layer/MatMul/ReadVariableOp"hidden_layer/MatMul/ReadVariableOp2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г

.__inference_hidden_layer_layer_call_fn_6221764

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221398p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
,
ц
G__inference_sequential_layer_call_and_return_conditional_losses_6221710

inputs?
+hidden_layer_matmul_readvariableop_resource:
;
,hidden_layer_biasadd_readvariableop_resource:	>
+output_layer_matmul_readvariableop_resource:	
:
,output_layer_biasadd_readvariableop_resource:

identityЂ#hidden_layer/BiasAdd/ReadVariableOpЂ"hidden_layer/MatMul/ReadVariableOpЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
"hidden_layer/MatMul/ReadVariableOpReadVariableOp+hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
hidden_layer/MatMulMatMulflatten/Reshape:output:0*hidden_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
#hidden_layer/BiasAdd/ReadVariableOpReadVariableOp,hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
hidden_layer/BiasAddBiasAddhidden_layer/MatMul:product:0+hidden_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ{
leaky_re_lu/LeakyRelu	LeakyReluhidden_layer/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
alpha%
з#<
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0 
output_layer/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

&unit_normalization/l2_normalize/SquareSquareoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

5unit_normalization/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:й
#unit_normalization/l2_normalize/SumSum*unit_normalization/l2_normalize/Square:y:0>unit_normalization/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(n
)unit_normalization/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+Ц
'unit_normalization/l2_normalize/MaximumMaximum,unit_normalization/l2_normalize/Sum:output:02unit_normalization/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%unit_normalization/l2_normalize/RsqrtRsqrt+unit_normalization/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
unit_normalization/l2_normalizeMuloutput_layer/BiasAdd:output:0)unit_normalization/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ђ
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity#unit_normalization/l2_normalize:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
NoOpNoOp$^hidden_layer/BiasAdd/ReadVariableOp#^hidden_layer/MatMul/ReadVariableOp6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2J
#hidden_layer/BiasAdd/ReadVariableOp#hidden_layer/BiasAdd/ReadVariableOp2H
"hidden_layer/MatMul/ReadVariableOp"hidden_layer/MatMul/ReadVariableOp2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
#
Ч
G__inference_sequential_layer_call_and_return_conditional_losses_6221478
flatten_input(
hidden_layer_6221457:
#
hidden_layer_6221459:	'
output_layer_6221463:	
"
output_layer_6221465:

identityЂ$hidden_layer/StatefulPartitionedCallЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЂ$output_layer/StatefulPartitionedCallЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpП
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6221382
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hidden_layer_6221457hidden_layer_6221459*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221398ч
leaky_re_lu/PartitionedCallPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221409Ё
$output_layer/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0output_layer_6221463output_layer_6221465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6221425є
"unit_normalization/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221442
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_6221457* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_layer_6221463*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+unit_normalization/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp%^hidden_layer/StatefulPartitionedCall6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameflatten_input
џ
Е
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221778

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О
`
D__inference_flatten_layer_call_and_return_conditional_losses_6221382

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ
Е
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221398

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
Г
I__inference_output_layer_layer_call_and_return_conditional_losses_6221425

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
#
Ч
G__inference_sequential_layer_call_and_return_conditional_losses_6221453
flatten_input(
hidden_layer_6221399:
#
hidden_layer_6221401:	'
output_layer_6221426:	
"
output_layer_6221428:

identityЂ$hidden_layer/StatefulPartitionedCallЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЂ$output_layer/StatefulPartitionedCallЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpП
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6221382
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hidden_layer_6221399hidden_layer_6221401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221398ч
leaky_re_lu/PartitionedCallPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221409Ё
$output_layer/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0output_layer_6221426output_layer_6221428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6221425є
"unit_normalization/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221442
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_6221399* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_layer_6221426*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+unit_normalization/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp%^hidden_layer/StatefulPartitionedCall6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameflatten_input
З
љ
#__inference__traced_restore_6221914
file_prefix8
$assignvariableop_hidden_layer_kernel:
3
$assignvariableop_1_hidden_layer_bias:	9
&assignvariableop_2_output_layer_kernel:	
2
$assignvariableop_3_output_layer_bias:


identity_5ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3щ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B З
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOpAssignVariableOp$assignvariableop_hidden_layer_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hidden_layer_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_2AssignVariableOp&assignvariableop_2_output_layer_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_3AssignVariableOp$assignvariableop_3_output_layer_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ќ

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
н	
М
__inference_loss_fn_0_6221836R
>hidden_layer_kernel_regularizer_l2loss_readvariableop_resource:

identityЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЖ
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>hidden_layer_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'hidden_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp
Ї
E
)__inference_flatten_layer_call_fn_6221749

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6221382a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н
к
,__inference_sequential_layer_call_fn_6221555
flatten_input
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6221544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameflatten_input
Ј
г
,__inference_sequential_layer_call_fn_6221676

inputs
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6221544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л	
Л
__inference_loss_fn_1_6221845Q
>output_layer_kernel_regularizer_l2loss_readvariableop_resource:	

identityЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpЕ
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>output_layer_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'output_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp
Н
к
,__inference_sequential_layer_call_fn_6221517
flatten_input
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6221506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameflatten_input
ю"
Р
G__inference_sequential_layer_call_and_return_conditional_losses_6221506

inputs(
hidden_layer_6221485:
#
hidden_layer_6221487:	'
output_layer_6221491:	
"
output_layer_6221493:

identityЂ$hidden_layer/StatefulPartitionedCallЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЂ$output_layer/StatefulPartitionedCallЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpИ
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6221382
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hidden_layer_6221485hidden_layer_6221487*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221398ч
leaky_re_lu/PartitionedCallPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221409Ё
$output_layer/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0output_layer_6221491output_layer_6221493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6221425є
"unit_normalization/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221442
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_6221485* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_layer_6221491*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+unit_normalization/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp%^hidden_layer/StatefulPartitionedCall6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю"
Р
G__inference_sequential_layer_call_and_return_conditional_losses_6221544

inputs(
hidden_layer_6221523:
#
hidden_layer_6221525:	'
output_layer_6221529:	
"
output_layer_6221531:

identityЂ$hidden_layer/StatefulPartitionedCallЂ5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpЂ$output_layer/StatefulPartitionedCallЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpИ
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6221382
$hidden_layer/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0hidden_layer_6221523hidden_layer_6221525*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221398ч
leaky_re_lu/PartitionedCallPartitionedCall-hidden_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221409Ё
$output_layer/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0output_layer_6221529output_layer_6221531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6221425є
"unit_normalization/PartitionedCallPartitionedCall-output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221442
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_6221523* 
_output_shapes
:
*
dtype0
&hidden_layer/kernel/Regularizer/L2LossL2Loss=hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%hidden_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#hidden_layer/kernel/Regularizer/mulMul.hidden_layer/kernel/Regularizer/mul/x:output:0/hidden_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_layer_6221529*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+unit_normalization/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp%^hidden_layer/StatefulPartitionedCall6^hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2L
$hidden_layer/StatefulPartitionedCall$hidden_layer/StatefulPartitionedCall2n
5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp5hidden_layer/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г,
Ћ
 __inference__traced_save_6221892
file_prefix>
*read_disablecopyonread_hidden_layer_kernel:
9
*read_1_disablecopyonread_hidden_layer_bias:	?
,read_2_disablecopyonread_output_layer_kernel:	
8
*read_3_disablecopyonread_output_layer_bias:

savev2_const

identity_9ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: |
Read/DisableCopyOnReadDisableCopyOnRead*read_disablecopyonread_hidden_layer_kernel"/device:CPU:0*
_output_shapes
 Ј
Read/ReadVariableOpReadVariableOp*read_disablecopyonread_hidden_layer_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
~
Read_1/DisableCopyOnReadDisableCopyOnRead*read_1_disablecopyonread_hidden_layer_bias"/device:CPU:0*
_output_shapes
 Ї
Read_1/ReadVariableOpReadVariableOp*read_1_disablecopyonread_hidden_layer_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_2/DisableCopyOnReadDisableCopyOnRead,read_2_disablecopyonread_output_layer_kernel"/device:CPU:0*
_output_shapes
 ­
Read_2/ReadVariableOpReadVariableOp,read_2_disablecopyonread_output_layer_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	
~
Read_3/DisableCopyOnReadDisableCopyOnRead*read_3_disablecopyonread_output_layer_bias"/device:CPU:0*
_output_shapes
 І
Read_3/ReadVariableOpReadVariableOp*read_3_disablecopyonread_output_layer_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
ц
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B А
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes	
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_8Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_9IdentityIdentity_8:output:0^NoOp*
T0*
_output_shapes
: Ѓ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*
_input_shapes
: : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
Љ
I
-__inference_leaky_re_lu_layer_call_fn_6221783

inputs
identityД
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221409a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
P
4__inference_unit_normalization_layer_call_fn_6221816

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221442`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
О
`
D__inference_flatten_layer_call_and_return_conditional_losses_6221755

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И$
А
"__inference__wrapped_model_6221372
flatten_inputJ
6sequential_hidden_layer_matmul_readvariableop_resource:
F
7sequential_hidden_layer_biasadd_readvariableop_resource:	I
6sequential_output_layer_matmul_readvariableop_resource:	
E
7sequential_output_layer_biasadd_readvariableop_resource:

identityЂ.sequential/hidden_layer/BiasAdd/ReadVariableOpЂ-sequential/hidden_layer/MatMul/ReadVariableOpЂ.sequential/output_layer/BiasAdd/ReadVariableOpЂ-sequential/output_layer/MatMul/ReadVariableOpi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
sequential/flatten/ReshapeReshapeflatten_input!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџІ
-sequential/hidden_layer/MatMul/ReadVariableOpReadVariableOp6sequential_hidden_layer_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0З
sequential/hidden_layer/MatMulMatMul#sequential/flatten/Reshape:output:05sequential/hidden_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЃ
.sequential/hidden_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_hidden_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0П
sequential/hidden_layer/BiasAddBiasAdd(sequential/hidden_layer/MatMul:product:06sequential/hidden_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu(sequential/hidden_layer/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
alpha%
з#<Ѕ
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0С
sequential/output_layer/MatMulMatMul.sequential/leaky_re_lu/LeakyRelu:activations:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ђ
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0О
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

1sequential/unit_normalization/l2_normalize/SquareSquare(sequential/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

@sequential/unit_normalization/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:њ
.sequential/unit_normalization/l2_normalize/SumSum5sequential/unit_normalization/l2_normalize/Square:y:0Isequential/unit_normalization/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(y
4sequential/unit_normalization/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+ч
2sequential/unit_normalization/l2_normalize/MaximumMaximum7sequential/unit_normalization/l2_normalize/Sum:output:0=sequential/unit_normalization/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџЃ
0sequential/unit_normalization/l2_normalize/RsqrtRsqrt6sequential/unit_normalization/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџУ
*sequential/unit_normalization/l2_normalizeMul(sequential/output_layer/BiasAdd:output:04sequential/unit_normalization/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
}
IdentityIdentity.sequential/unit_normalization/l2_normalize:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp/^sequential/hidden_layer/BiasAdd/ReadVariableOp.^sequential/hidden_layer/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2`
.sequential/hidden_layer/BiasAdd/ReadVariableOp.sequential/hidden_layer/BiasAdd/ReadVariableOp2^
-sequential/hidden_layer/MatMul/ReadVariableOp-sequential/hidden_layer/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameflatten_input
Ј
г
,__inference_sequential_layer_call_fn_6221663

inputs
unknown:

	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6221506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
Г
I__inference_output_layer_layer_call_and_return_conditional_losses_6221811

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

5output_layer/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
&output_layer/kernel/Regularizer/L2LossL2Loss=output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ќ
#output_layer/kernel/Regularizer/mulMul.output_layer/kernel/Regularizer/mul/x:output:0/output_layer/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp6^output_layer/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp5output_layer/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А	
k
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221827

inputs
identityW
l2_normalize/SquareSquareinputs*
T0*'
_output_shapes
:џџџџџџџџџ
l
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЬМ+
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџg
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџe
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
X
IdentityIdentityl2_normalize:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ
:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ц
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221788

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:џџџџџџџџџ*
alpha%
з#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Х
serving_defaultБ
K
flatten_input:
serving_default_flatten_input:0џџџџџџџџџF
unit_normalization0
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:Д
Ь
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias"
_tf_keras_layer
Џ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0axis"
_tf_keras_layer
<
0
1
(2
)3"
trackable_list_wrapper
<
0
1
(2
)3"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
Ъ
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
л
8trace_0
9trace_1
:trace_2
;trace_32№
,__inference_sequential_layer_call_fn_6221517
,__inference_sequential_layer_call_fn_6221555
,__inference_sequential_layer_call_fn_6221663
,__inference_sequential_layer_call_fn_6221676Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z8trace_0z9trace_1z:trace_2z;trace_3
Ч
<trace_0
=trace_1
>trace_2
?trace_32м
G__inference_sequential_layer_call_and_return_conditional_losses_6221453
G__inference_sequential_layer_call_and_return_conditional_losses_6221478
G__inference_sequential_layer_call_and_return_conditional_losses_6221710
G__inference_sequential_layer_call_and_return_conditional_losses_6221744Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z<trace_0z=trace_1z>trace_2z?trace_3
гBа
"__inference__wrapped_model_6221372flatten_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,
@serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
у
Ftrace_02Ц
)__inference_flatten_layer_call_fn_6221749
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zFtrace_0
ў
Gtrace_02с
D__inference_flatten_layer_call_and_return_conditional_losses_6221755
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zGtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
10"
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ш
Mtrace_02Ы
.__inference_hidden_layer_layer_call_fn_6221764
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zMtrace_0

Ntrace_02ц
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221778
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zNtrace_0
':%
2hidden_layer/kernel
 :2hidden_layer/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ч
Ttrace_02Ъ
-__inference_leaky_re_lu_layer_call_fn_6221783
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zTtrace_0

Utrace_02х
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221788
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zUtrace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
'
20"
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ш
[trace_02Ы
.__inference_output_layer_layer_call_fn_6221797
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z[trace_0

\trace_02ц
I__inference_output_layer_layer_call_and_return_conditional_losses_6221811
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z\trace_0
&:$	
2output_layer/kernel
:
2output_layer/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ю
btrace_02б
4__inference_unit_normalization_layer_call_fn_6221816
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zbtrace_0

ctrace_02ь
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221827
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zctrace_0
 "
trackable_list_wrapper
Ю
dtrace_02Б
__inference_loss_fn_0_6221836
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zdtrace_0
Ю
etrace_02Б
__inference_loss_fn_1_6221845
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zetrace_0
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њBї
,__inference_sequential_layer_call_fn_6221517flatten_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
,__inference_sequential_layer_call_fn_6221555flatten_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
,__inference_sequential_layer_call_fn_6221663inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
,__inference_sequential_layer_call_fn_6221676inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_6221453flatten_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_6221478flatten_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_6221710inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_6221744inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
вBЯ
%__inference_signature_wrapper_6221642flatten_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
гBа
)__inference_flatten_layer_call_fn_6221749inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_flatten_layer_call_and_return_conditional_losses_6221755inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_hidden_layer_layer_call_fn_6221764inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221778inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
зBд
-__inference_leaky_re_lu_layer_call_fn_6221783inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221788inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_output_layer_layer_call_fn_6221797inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_output_layer_layer_call_and_return_conditional_losses_6221811inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
оBл
4__inference_unit_normalization_layer_call_fn_6221816inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221827inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ДBБ
__inference_loss_fn_0_6221836"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ДBБ
__inference_loss_fn_1_6221845"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ В
"__inference__wrapped_model_6221372():Ђ7
0Ђ-
+(
flatten_inputџџџџџџџџџ
Њ "GЊD
B
unit_normalization,)
unit_normalizationџџџџџџџџџ
Ќ
D__inference_flatten_layer_call_and_return_conditional_losses_6221755d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
)__inference_flatten_layer_call_fn_6221749Y3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџВ
I__inference_hidden_layer_layer_call_and_return_conditional_losses_6221778e0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
.__inference_hidden_layer_layer_call_fn_6221764Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ­
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6221788a0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
-__inference_leaky_re_lu_layer_call_fn_6221783V0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџE
__inference_loss_fn_0_6221836$Ђ

Ђ 
Њ "
unknown E
__inference_loss_fn_1_6221845$(Ђ

Ђ 
Њ "
unknown Б
I__inference_output_layer_layer_call_and_return_conditional_losses_6221811d()0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
.__inference_output_layer_layer_call_fn_6221797Y()0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
У
G__inference_sequential_layer_call_and_return_conditional_losses_6221453x()BЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 У
G__inference_sequential_layer_call_and_return_conditional_losses_6221478x()BЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 М
G__inference_sequential_layer_call_and_return_conditional_losses_6221710q();Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 М
G__inference_sequential_layer_call_and_return_conditional_losses_6221744q();Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
,__inference_sequential_layer_call_fn_6221517m()BЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџ

,__inference_sequential_layer_call_fn_6221555m()BЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџ

,__inference_sequential_layer_call_fn_6221663f();Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџ

,__inference_sequential_layer_call_fn_6221676f();Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџ
Ц
%__inference_signature_wrapper_6221642()KЂH
Ђ 
AЊ>
<
flatten_input+(
flatten_inputџџџџџџџџџ"GЊD
B
unit_normalization,)
unit_normalizationџџџџџџџџџ
В
O__inference_unit_normalization_layer_call_and_return_conditional_losses_6221827_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
4__inference_unit_normalization_layer_call_fn_6221816T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ

Њ "!
unknownџџџџџџџџџ
