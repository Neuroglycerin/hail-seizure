# coding: utf-8



def main():
    """
    Contains the main loop for this script.
    Pseudo-MHMCMC to find optimal AUC scoring 
    combinations of HDF5s.
    """
    # pseudo-code for the MCMC iteration
    # want it to start with the probably good features
    # Sample a new hdf5 and replace existing at random
    #   Or, just push it in, or just drop a hdf5 at random
    # Evaluate the AUC for this, run train.py
    # compute acceptance probability from AUC:
    #     r = min(1,AUC/(previous AUC))
    # accept new point with probability r

    # Should we use log loss instead of AUC to maintain probabilistic interpretation?
    # Not so familiar with log loss.
    # Seems this method will at least search for optimum AUC scores, which is the aim.


for i in range(20):
    newjson = {}
    newjson.update(start)
    features = start['FEATURES']
    shuffle(features)
    shuffle(promising)
    features[0] = promising[0]
    with open('{0}_swap_{1}.json'.format(features[0],promising[0]),'w') as f:
        newjson['FEATURES']=features
        json.dump(newjson,f)


if __name__ == "__main__":
    main()
