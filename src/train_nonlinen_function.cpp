//============================================================================
// Name        : train_nonlinen_function.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#undef FIXEDFANN
//#include <fann.h>
#include <floatfann.h>
#include <string.h>
using namespace std;

#define NEURONES_HIDDEN (num_input/2)

typedef double(*function_type)( double );

inline double function_x_pow_3( double x )
{
	return x*x*x;
}

inline double function_2_mul_x( double x )
{
	return 2 * x;
}

inline double function_x_sin_x( double x )
{
	return (rand() % 20) * sin( 2 * (rand()%20) * M_PI* x );
}

inline double function_random( double x )
{
	return ( rand() % (int)x );
}
/*
 * Creates an empty set of training data
 */
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train_my(unsigned int num_data, unsigned int num_input, unsigned int num_output);

inline double delta( double precision )
{
	double value = 0.0;
	value = ((double)1 / precision);
	return value;
}

fann_train_data * prepeare_train_data( unsigned int num_data, unsigned int num_input, unsigned int num_output, double start_number, function_type function, unsigned int output_true_index )
{
	fann_train_data * train_data = fann_create_train_my( num_data, num_input, num_output );

	srand( time( 0 ) );

	double current_number = start_number;

	for( unsigned int data_i = 0 ; data_i <  train_data->num_data ; data_i++ )
	{
		current_number = current_number + rand() % num_input;

		for( unsigned int input_i = 0 ; input_i <  train_data->num_input ; input_i++ )
		{
			double x = current_number + 1 * input_i;
			double y = function( x );

			train_data->input[data_i][input_i] = y;
		}
		fann_type * output = train_data->output[data_i];
		memset( output, 0, num_output );
		train_data->output[data_i][output_true_index] = 1;
	}

	return train_data;
}

int main()
{
	const unsigned int num_input_layers = 1;
	const unsigned int num_output_layers = 1;
	const unsigned int num_input = 100;
	const unsigned int num_output = 4;
	const unsigned int num_data = 100;
	const unsigned int num_neurons_hidden[] = { NEURONES_HIDDEN };
	const unsigned int num_hidden_layers = sizeof(num_neurons_hidden)/sizeof(num_neurons_hidden[0]);
	const unsigned int num_layers = num_input_layers + num_hidden_layers + num_output_layers ;
	const float desired_error = (const float) 0.005;
	const unsigned int max_epochs = 5000;
	const unsigned int epochs_between_reports = 200;

	fann_train_data * train_data_x3 = prepeare_train_data( num_data, num_input, num_output, (double)1, function_x_pow_3, 0);
	fann_train_data * test_data_x3 = prepeare_train_data( num_data, num_input, num_output, (double)5, function_x_pow_3, 0);

	fann_train_data * train_data_2x = prepeare_train_data( num_data, num_input, num_output, (double)1, function_2_mul_x, 1);
	fann_train_data * test_data_2x = prepeare_train_data( num_data, num_input, num_output, (double)10, function_2_mul_x, 1);

    fann_train_data * train_data_zero = prepeare_train_data( num_data, num_input, num_output, (double)1, function_x_sin_x, 2);
    fann_train_data * test_data_zero = prepeare_train_data( num_data, num_input, num_output, (double)1000, function_x_sin_x, 2);

    fann_train_data * train_data_random = prepeare_train_data( num_data, num_input, num_output, (double)100, function_random, 3);
	fann_train_data * test_data_random = prepeare_train_data( num_data, num_input, num_output, (double)100, function_random, 3);

	fann_train_data * train_data_freqs = fann_read_train_from_file( "./freqs_train.txt" );

	struct fann *ann = fann_create_standard(num_layers, train_data_freqs->num_input, NEURONES_HIDDEN, train_data_freqs->num_output);
////////SCALE TEST DATA
//	fann_scale_input_train_data (train_data_x3, 0.0, 1.0);
//	fann_scale_input_train_data (test_data_x3, 0.0, 1.0);
//	fann_set_input_scaling_params( ann, train_data_x3, 0.0, 1.0);
//
//	fann_scale_input_train_data (train_data_2x, 0.0, 1.0);
//	fann_scale_input_train_data (test_data_2x, 0.0, 1.0);
//	fann_set_input_scaling_params( ann, train_data_2x, 0.0, 1.0);
//
//	fann_scale_input_train_data (train_data_zero, 0.0, 1.0);
//	fann_scale_input_train_data (test_data_zero, 0.0, 1.0);
//	fann_set_input_scaling_params( ann, train_data_zero, 0.0, 1.0);
//
//	fann_scale_input_train_data (train_data_random, 0.0, 1.0);
//	fann_scale_input_train_data (test_data_random, 0.0, 1.0);
//	fann_set_input_scaling_params( ann, train_data_random, 0.0, 1.0);

	fann_scale_input_train_data (train_data_freqs, 0.0, 1.0);
	fann_set_input_scaling_params( ann, train_data_freqs, 0.0, 1.0);
////////

	fann_save_train(train_data_x3, "x3.train");
	fann_save_train(train_data_2x, "2x.train");
	fann_save_train(train_data_zero, "zero.train");
	fann_save_train(train_data_random, "random.train");
	fann_save_train(test_data_x3, "x3_test.train");

	fann_train_data * train_data = 0;

	fann_train_data * merged_train_data = fann_merge_train_data( train_data_x3, train_data_2x );

	merged_train_data = fann_merge_train_data( merged_train_data, train_data_zero );
	merged_train_data = fann_merge_train_data( merged_train_data, train_data_random );

//	fann_shuffle_train_data( merged_train_data );

	fann_save_train( merged_train_data, "merged.train" );


//	fann_init_weights(ann, train_data);
//	fann_randomize_weights( ann, 0, 1 );
//	fann_set_learning_rate(ann, 0.05);
//	fann_set_activation_steepness_hidden(ann, 0.5);
//        fann_set_activation_steepness_output(ann, 1.0);
//        fann_set_activation_function_hidden(ann, FANN_ELLIOT);
//      fann_set_activation_function_output(ann, FANN_SIGMOID);
//	fann_set_training_algorithm( ann, FANN_TRAIN_INCREMENTAL );

//	fann_train(ann, train_data->input[0], train_data->output[0]);
//	fann_train(ann, train_data->input[1], train_data->output[1]);
//	fann_train(ann, train_data->input[2], train_data->output[2]);
//	fann_train(ann, train_data->input[3], train_data->output[3]);


/*
	train_data = train_data_x3;/////////SET TRAIN DATA
	for( int i = 0 ; i < train_data->num_data ; i++ )
	{
		fann_train( ann, train_data_x3->input[i], train_data_x3->output[i] );
		fann_train( ann, train_data_2x->input[i], train_data_2x->output[i] );
		fann_train( ann, train_data_zero->input[i], train_data_zero->output[i] );
//		fann_train( ann, train_data->input[i], train_data->output[i] );
	}
        train_data = train_data_2x;/////////SET TRAIN DATA
	for( int i = 0 ; i < train_data->num_data ; i++ )
        {
                fann_train( ann, train_data_x3->input[i], train_data_x3->output[i] );
                fann_train( ann, train_data_2x->input[i], train_data_2x->output[i] );
                fann_train( ann, train_data_zero->input[i], train_data_zero->output[i] );
//		fann_train( ann, train_data->input[i], train_data->output[i] );
        }
*/
	for( unsigned int epoch_i = 0 ; epoch_i < 100 ; epoch_i++ )
	{
//		train_data = train_data_zero;/////////SET TRAIN DATA
		for( int data_i = 0 ; data_i < num_data ; data_i++ )
        	{
        //        	fann_train( ann, train_data_x3->input[data_i], train_data_x3->output[data_i] );
	//                fann_train( ann, train_data_2x->input[data_i], train_data_2x->output[data_i] );
        //	        fann_train( ann, train_data_zero->input[data_i], train_data_zero->output[data_i] );
//			fann_train( ann, train_data->input[data_i], train_data->output[data_i] );
        	}
		cout<<"*"<<flush;
	}


//	fann_set_input_scaling_params(ann, train_data_x3, 0, 1);
//	fann_set_output_scaling_params(ann, train_data_x3, 0, 1);

	fann_train_on_data( ann, train_data_freqs, max_epochs, epochs_between_reports, desired_error);
//
//    for( unsigned int i = 0 ; i < 10000 ; i++ )
//    {
//    	for( unsigned int j = 0 ; j < train_data_x3->num_data ; j++ )
//    	{
//    		fann_train(ann, train_data->input[j], train_data->output[j]);
//    	}
//    }

//	for(unsigned int i = 0; i < train_data->num_data; i++)
//	{
//		fann_reset_MSE(ann);
//		calc_out = fann_test(ann, train_data->input[i], train_data->output[i]);
//#ifdef FIXEDFANN
//		printf("XOR test (%d, %d) -> %d, should be %d, difference=%f\n",
//				train_data->input[i][0], train_data->input[i][1], calc_out[0], train_data->output[i][0],
//			   (float) fann_abs(calc_out[0] - train_data->output[i][0]) / fann_get_multiplier(ann));
//
//		if((float) fann_abs(*calc_out - train_data->output[i][0]) / fann_get_multiplier(ann) > 0.1)
//		{
//			printf("Test failed\n");
//		}
//#else
//		printf("XOR test (%f, %f) -> %f, should be %f, difference=%f\n",
//			   train_data->input[i][0], train_data->input[i][1], *calc_out, train_data->output[i][0],
//			   (float) fann_abs(*calc_out - train_data->output[i][0]));
//#endif
//	}


	{
//		fann_set_activation_function_hidden(ann, FANN_LINEAR);
//
//		fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);

		fann_save(ann, "../and.net");

		printf("training_algorithm=%d\n", ann->training_algorithm);
	}

	train_data = train_data_freqs;/////////SET TRAIN DATA
	for( unsigned int data_i = 0 ; data_i <  train_data->num_data ; data_i++ )
	{
			fann_type * desired_ouput = new fann_type;
			* desired_ouput = 1;
			fann_type * calc_out = fann_run( ann, train_data->input[data_i] );
			fann_type * desired = train_data->output[data_i];
			printf("desired [%d]=%f, %f, %f, %f, %f, %f, %f\n", data_i, desired[0], desired[1], desired[2], desired[3], desired[4], desired[5], desired[6]);fflush(stdout);
			printf("test    [%d]=%f, %f, %f, %f, %f, %f, %f\n", data_i, calc_out[0], calc_out[1], calc_out[2], calc_out[3], calc_out[4], calc_out[5], calc_out[6]);fflush(stdout);
	}

	float mse = fann_test_data( ann, train_data );
	unsigned int bitfail = fann_get_bit_fail( ann );

	fann_destroy_train(train_data);
	fann_destroy(ann);

	return 0;
}

/*
 * Creates an empty set of training data
 */
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train_my(unsigned int num_data, unsigned int num_input, unsigned int num_output)
{
	fann_type *data_input, *data_output;
	unsigned int i;
	struct fann_train_data *data =
		(struct fann_train_data *) malloc(sizeof(struct fann_train_data));

	if(data == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	fann_init_error_data((struct fann_error *) data);

	data->num_data = num_data;
	data->num_input = num_input;
	data->num_output = num_output;
	data->input = (fann_type **) calloc(num_data, sizeof(fann_type *));
	if(data->input == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data->output = (fann_type **) calloc(num_data, sizeof(fann_type *));
	if(data->output == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data_input = (fann_type *) calloc(num_input * num_data, sizeof(fann_type));
	if(data_input == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data_output = (fann_type *) calloc(num_output * num_data, sizeof(fann_type));
	if(data_output == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	for(i = 0; i != num_data; i++)
	{
		data->input[i] = data_input;
		data_input += num_input;
		data->output[i] = data_output;
		data_output += num_output;
	}
	return data;
}
