// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


#ifdef GPU
void pull_network_output(network *net);
#endif

char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);


int get_predicted_class_network(network *net);
void print_network(network *net);
void calc_network_cost(network *net);

#endif

