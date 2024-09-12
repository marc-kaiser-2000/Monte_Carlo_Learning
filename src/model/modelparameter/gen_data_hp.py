import tensorflow as tf
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter


class GenData(AbstractHyperparameter):
    """Class implements configurations of the 'gen_data' and 'mc_step' function.
    Different Approaches are defined - but only approaches 4 - 6 shall be used.
    The other approaches are exist only for reference.

    Configuration 4 implements a default Milstein solver.
    Configuration 5 implements a Milstein solver with a control variate.
    Configuration 6 implements a default Euler solver.
    """

    def __init__(self,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)

        self._parameter_value_gen_data_l0 = None
        self._parameter_value_gen_data_ln = None
        self._parameter_value_mc_step_l0 = None
        self._parameter_value_mc_step_ln = None

        self._set_gen_data_parameter_value(conf)
        self._set_mc_step_parameter_value(conf)

    def get_active_value(self):
        return NotImplementedError("Class 'GenData' has two '_parameter_values' use specific getters!")
    
    def get_parameter_value(self):
        return NotImplementedError("Class 'GenData' can not return the '_parameter_value' since it is split into several values!")

    def get_active_value_gen_data(self,level):
        """Gets a pointer to 'gen_data' configuration

        Args:
            level (int): Level of trained neural network.

        Returns:
            *fnc : Function pointer to selected 'gen_data' configuration
        """
        if level == 0:
            return self._parameter_value_gen_data_l0
        else:
            return self._parameter_value_gen_data_ln

    def get_active_value_mc_step(self,level):
        """Gets a pointer to 'mc_step' configuration

        Args:
            level (int): Level of trained neural network.

        Returns:
            *fnc : Function pointer to selected 'mc_step' configuration
        """
        if level == 0:
            return self._parameter_value_mc_step_l0
        else:
            return self._parameter_value_mc_step_ln

    def _set_gen_data_parameter_value(self,conf):
        l0_gen_data_options = {
            4:gen_data_milstein_default_l0,
            5:gen_data_milstein_control_variate_l0,
            6:gen_data_euler_default_l0,
        }
        self._parameter_value_gen_data_l0=  l0_gen_data_options.get(conf)

        ln_gen_data_options = {
            4:gen_data_milstein_default_ln,
            5:gen_data_milstein_control_variate_ln, #5:gen_data_milstein_default_ln,
            6:gen_data_euler_default_ln,
        }

        self._parameter_value_gen_data_ln= ln_gen_data_options.get(conf)

    

    def _set_mc_step_parameter_value(self,conf):


        l0_mc_step_options = {
            4:mc_step_milstein_default_l0,
            5:mc_step_milstein_control_variate_l0,
            6:mc_step_euler_default_l0,
        }

        self._parameter_value_mc_step_l0 = l0_mc_step_options.get(conf)


        ln_mc_step_options = {
            4:mc_step_milstein_default_ln,
            5:mc_step_milstein_control_variate_ln, #5:mc_step_milstein_default_ln,
            6:mc_step_euler_default_ln,
        }

        self._parameter_value_mc_step_ln =  ln_mc_step_options.get(conf)



#================================================================================
# Configuration 1: Time and Space Complexity Optimized (Default Milstein Method)
#================================================================================

@tf.function
def fun_g_space_time(const,x):
    #print("Function fun_g_space_time: tracing!")
    return tf.exp(-const.r*const.T) * tf.maximum(tf.reduce_sum(x/const.dim, axis=2, keepdims=True) - const.K, 0.)

@tf.function
def mc_step(const,X,y,level,n_points,mc_samples,num_timestep):
    #print("Function mc_step: tracing!")

    # Initialize Variables
    y_sample = []
    dt = const.T/num_timestep

    samplers = []
    samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, const.dim), dtype=const.DTYPE))
    y_sample.append(0.)

    # Only create second sampler for cosarity levels != 0
    if level != 0:
        samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,const.dim)),axis = 1))
        y_sample.append(0.)

    # Iterate over samplers; first fine the coarse
    for k in range(2):
        ST = X

        # Skip second iteration if Level 0
        if level == 0 and k != 0:
            continue
        dt =  const.T/samplers[k].shape[0]

        # Iteratve over number of timesteps
        for i in range(samplers[k].shape[0]): 
            Si = tf.matmul(samplers[k][i,:,:], const.C) 
            ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

        # Update output
        ST_expanded  =  tf.expand_dims(ST,axis = 0)           
        ST_fung = fun_g_space_time(const,ST_expanded)
        y_sample[k] = tf.squeeze(ST_fung,axis=0)/mc_samples

    # Calculate Difference for levels greater 0
    if level != 0:
        y += (y_sample[0] - y_sample[1])

    else:
        y += (y_sample[0])

    return y


@tf.function
def gen_data_opt_time(const,level,n_points,mc_samples,num_timestep):
    #print("Function gen_data_opt_time: tracing!")
    y_sample = [0.,0.]
    dt =  const.T/num_timestep
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)


    samplers = []
    samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points,mc_samples, const.dim), dtype=const.DTYPE))
    # Only create second sampler for cosarity levels != 0
    if level != 0:
        samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,mc_samples,const.dim)),axis = 1))


    # Iterate over samplers; first fine the coarse
    for k in range(2):
        ST = tf.tile(tf.expand_dims(S0,axis = 1),[1,mc_samples,1])

        # Skip second iteration if Level 0
        if level == 0 and k != 0:
            continue
        
        dt =  const.T/samplers[k].shape[0]

        # Iteratve over number of timesteps
        for i in range(samplers[k].shape[0]): 
            Si = tf.matmul(samplers[k][i,:,:,:], const.C)
            ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

        # Update output
        y_sample[k] =  tf.reduce_sum(fun_g_space_time(const,ST), axis=1,keepdims=False)/(mc_samples)

    y = y_sample[0] - y_sample[1]

    return S0, y

@tf.function
def gen_data_opt_space(const,level,n_points,mc_samples,num_timestep):
    #print("Function gen_data_opt_space: tracing!")
    # Initialize Variables
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)
    y_sample = [0.,0.]

    for j in range(mc_samples):

        dt =  const.T/num_timestep
        samplers = []
        samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, const.dim), dtype=const.DTYPE))


        # Only create second sampler for cosarity levels != 0
        if level != 0:
            samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,const.dim)),axis = 1))

        # Iterate over samplers; first fine the coarse
        for k in range(2):
            #ST = tf.tile(tf.expand_dims(S0,axis = 1),[1,mc_samples,1])
            ST = S0

            # Skip second iteration if Level 0
            if level == 0 and k != 0:
                continue

            dt =  const.T/samplers[k].shape[0]

            # Iteratve over number of timesteps
            for i in range(samplers[k].shape[0]): 
                #Si = tf.matmul(samplers[k][i,:,:,:], const.C)
                Si = tf.matmul(samplers[k][i,:,:], const.C)
                ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

            # Update output
            ST_expanded  =  tf.expand_dims(ST,axis = 0)           
            ST_fung = fun_g_space_time(const,ST_expanded)
            y_sample[k] += tf.squeeze(ST_fung,axis=0)/(mc_samples)

    y = y_sample[0] - y_sample[1]

    return S0, y

#================================================================================
# Configuration 2: Tensorflow Constants (Default Milstein Method)
#================================================================================

DTYPE_GLOBAL = tf.float32
DTYPE_ITERATOR = tf.int32
idx0 = tf.constant(0,dtype=tf.int32)
idx1 = tf.constant(1,dtype=tf.int32)

@tf.function
def fun_g_tf_const(x):
        #print("Function fun_g_tf_const: tracing!")
        r = get_constant_interest_rate_drift()
        T = get_constant_final_time()
        K = get_constant_strike_price()
        #tf.print(tf.shape(x)[2])
        #tf.print(dim)
        return tf.exp(-r*T) * tf.maximum(tf.reduce_sum(x/tf.cast(tf.shape(x)[2],dtype=DTYPE_GLOBAL), axis=2, keepdims=True) - K, 0.)
#return tf.exp(-const.r*const.T) * tf.maximum(tf.reduce_sum(x/const.dim, axis=2, keepdims=True) - const.K, 0.)


@tf.function
def gen_data_tf_const(level,n_points,mc_samples,num_timestep,dim):
    #print("Function gen_data_tf_const: tracing!")
    ST_fine = tf.zeros(
            shape=(n_points,dim),
            dtype=DTYPE_GLOBAL
            )

    S0 = init_S0_random_normal(n_points,dim)

    y_sample = tf.TensorArray(dtype=DTYPE_GLOBAL,size=0,dynamic_size=True,clear_after_read=False)
    y_sample = y_sample.write(0,tf.zeros(shape=(n_points,1),dtype=DTYPE_GLOBAL))
    
    if level != tf.constant(0,dtype=tf.int32):
        y_sample = y_sample.write(1,tf.zeros(shape=(n_points,1),dtype=DTYPE_GLOBAL))


    r = get_constant_interest_rate_drift()
    sigma = get_constant_diff_coefficient(dim)
    C = get_constant_asset_correlation(dim)

    for j in tf.range(mc_samples):
        if (j + tf.constant(1,dtype=tf.int32)) % tf.constant(10000,dtype=tf.int32) == tf.constant(0,dtype=tf.int32):
            tf.print(j + tf.constant(1,dtype=tf.int32), " / ", mc_samples)


        T = get_constant_final_time()
        dt = T/tf.cast(num_timestep,dtype=DTYPE_GLOBAL)

        samplers = tf.TensorArray(dtype=DTYPE_GLOBAL,size=0,dynamic_size=True,clear_after_read=False)
        samplers = samplers.write(0,tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, dim), dtype=DTYPE_GLOBAL))

        if level != tf.constant(0,dtype=tf.int32):  
            samplers = samplers.write(1,tf.reduce_sum(tf.reshape(samplers.read(0),shape = (int(num_timestep/2),2,n_points,dim)),axis = 1))


        for k in tf.range(tf.constant(2,dtype=tf.int32)):

            ST = S0

            if level == tf.constant(0,dtype=tf.int32) and k != tf.constant(0,dtype=tf.int32):
                continue

            dt = T/tf.cast(tf.shape(samplers.read(k))[0],dtype=DTYPE_GLOBAL)

            for i in tf.range(tf.shape(samplers.read(k))[0]):
                Si = tf.matmul(samplers.read(k)[i,:,:], C)
                ST = ST + ST * r * dt + tf.matmul(ST, sigma) * Si + 0.5 * tf.matmul(ST, sigma**2) * (Si**2 - dt)

            if k == tf.constant(0,dtype=tf.int32):
                ST_fine = ST_fine + (ST / tf.cast(mc_samples,dtype=DTYPE_GLOBAL))

            ST_expanded  =  tf.expand_dims(ST,axis = 0)           
            ST_fung = fun_g_tf_const(ST_expanded)

            y_sample = y_sample.write(k, y_sample.read(k) + (tf.squeeze(ST_fung,axis=0)/(tf.cast(mc_samples,dtype=DTYPE_GLOBAL))))

    if level != tf.constant(0,dtype=tf.int32):
        y = y_sample.read(0) - y_sample.read(1)
    else:
        y = y_sample.read(0)

    #X = tf.stack([S0, ST_fine], 2)
    X = S0
    return S0, y


@tf.function
def init_S0(Type,n_points,dim):
    #print("Function init_S0: tracing!")
    if Type == tf.constant(1,dtype=tf.int32):
        ngrid = tf.cast(n_points,dtype=DTYPE_GLOBAL)
        ngrid = tf.sqrt(ngrid)
        ngrid = tf.cast(ngrid,dtype=tf.int32)
        ngrid = ngrid -tf.constant(1,dtype=tf.int32)
        S0 = init_S0_xgrid(ngrid,dim)
        return S0

    else:
        S0 = init_S0_random_normal(n_points,dim)
        return S0

@tf.function
def init_S0_random_normal(n_points,dim):
    #print("Function init_S0_random_normal: tracing!")
    a, b = get_constant_domain_of_interest(dim)
    S0 = a + tf.random.uniform((n_points,dim), dtype=DTYPE_GLOBAL) * (b-a)
    return S0

@tf.function
def init_S0_xgrid(ngrid,dim):
    #print("Function init_S0_xgrid: tracing!")
    a, b = get_constant_domain_of_interest(dim)

    idx_all = tf.range(dim)
    idx = tf.stack([idx0,idx1],axis=0)

    expanded_idx_all = tf.expand_dims(idx_all,axis=0)
    expanded_idx = tf.expand_dims(idx,axis=0)


    idx_remaining = tf.sets.difference(expanded_idx_all,expanded_idx)
    idx_remaining = tf.sparse.to_dense(idx_remaining)
    idx_remaining = tf.squeeze(idx_remaining)

    values_all = tf.reduce_mean(tf.stack([a,b],axis=1),axis=1)


    x_space = tf.linspace(a[idx0], b[idx0], ngrid + tf.constant(1,dtype=tf.int32))
    y_space = tf.linspace(a[idx1], b[idx1], ngrid + tf.constant(1,dtype=tf.int32))

    X,Y = tf.meshgrid(x_space,y_space)


    values_remaining = tf.gather(values_all,idx_remaining)
    values_remaining =tf.expand_dims(values_remaining,axis=0)
    values_remaining = tf.tile(values_remaining,[tf.pow(ngrid + tf.constant(1,dtype=tf.int32),tf.constant(2,dtype=tf.int32)),1])

    X_flat = tf.expand_dims(tf.reshape(X,[-1]),axis=1)
    Y_flat = tf.expand_dims(tf.reshape(Y,[-1]),axis=1)

    Xgrid = tf.concat([X_flat,Y_flat,values_remaining],axis=1)
    Xgrid = tf.cast(Xgrid,dtype =DTYPE_GLOBAL)
    return Xgrid


@tf.function
def get_constant_final_time():
    #print("Function get_constant_final_time: tracing!")
    return tf.constant(1., dtype=DTYPE_GLOBAL)

@tf.function(input_signature=(tf.TensorSpec(shape=None, dtype=tf.int32),))
def get_constant_domain_of_interest(dim):
    #print("Function get_constant_domain_of_interest: tracing!")
    a = 90 * tf.ones((dim), dtype=DTYPE_GLOBAL)
    b = 110 * tf.ones((dim), dtype=DTYPE_GLOBAL)
    return a,b

@tf.function
def get_constant_interest_rate_drift():
    #print("Function get_constant_interest_rate_drift: tracing!")
    return tf.constant(1./20, dtype=DTYPE_GLOBAL)

@tf.function
def get_constant_strike_price():
    #print("Function get_constant_strike_price: tracing!")
    return tf.constant(100.,dtype=DTYPE_GLOBAL)

@tf.function(input_signature=(tf.TensorSpec(shape=None, dtype=tf.int32),))
def get_constant_diff_coefficient(dim):
    #print("Function get_constant_diff_coefficient: tracing!")
    return tf.linalg.diag(1./10 + 1./200*tf.range(1, dim+1, dtype=DTYPE_GLOBAL))

@tf.function(input_signature=(tf.TensorSpec(shape=None, dtype=tf.int32),))
def get_constant_asset_correlation(dim):
    #print("Function get_constant_asset_correlation: tracing!")
    Sig = tf.eye(dim) + 0.25 * (tf.ones((dim,dim)) - tf.eye(dim))
    C = tf.transpose(tf.linalg.cholesky(Sig))
    return C

#================================================================================
# Configuration 3: Python Constants (Default Milstein Method)
#================================================================================


@tf.function
def fun_g_python_const(const,x):
        #print("Function fun_g_python_const: tracing!")
        return tf.exp(-const.r*const.T) * tf.maximum(tf.reduce_sum(x/const.dim, axis=2, keepdims=True) - const.K, 0.)

@tf.function
def gen_data_python_const(const,level,n_points,mc_samples,num_timestep):
    #print("Function gen_data_python_const: tracing!")

    # Initialize Variables    
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)

    for j in range(mc_samples):
        y= mc_step_python_const(S0,const,level,n_points,mc_samples,num_timestep,y)
    return S0, y


@tf.function
def mc_step_python_const(init_S0,const,level,n_points,mc_samples,num_timestep,y):
    #print("Function mc_step: tracing!")

    # Initialize Variables
    y_sample = [0.,0.]
    dt = const.T/num_timestep

    samplers = []
    samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, const.dim), dtype=const.DTYPE))

    # Only create second sampler for cosarity levels != 0
    if level != 0:
        samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,const.dim)),axis = 1))

    for k in range(2):
        ST = init_S0

        # Skip second iteration if Level 0
        if level == 0 and k != 0:
            continue

        dt =  const.T/samplers[k].shape[0]

        # Iteratve over number of timesteps
        for i in range(samplers[k].shape[0]): 
            Si = tf.matmul(samplers[k][i,:,:], const.C) 
            ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

        # Update output
        ST_expanded  =  tf.expand_dims(ST,axis = 0)           
        ST_fung = fun_g_python_const(const,ST_expanded)
        y_sample[k] = tf.squeeze(ST_fung,axis=0)/mc_samples

    # Calculate Difference for levels greater 0
    y += (y_sample[0] - y_sample[1])

    return y


#================================================================================
# Configuration 4: Default Milstein
#================================================================================

@tf.function(jit_compile=True)
def fun_g_milstein_default(const,x):
    #print("Function fun_g_milstein_default: tracing!")  
    return tf.exp(-const.r*const.T) * tf.maximum(tf.reduce_sum(x/const.dim, axis=1, keepdims=True) - const.K, 0.)

@tf.function(jit_compile=True)
def gen_data_milstein_default_l0(const,n_points,mc_samples,num_timestep):
    #print("Function gen_data_milstein_default_l0: tracing!")
    # Initialize Variables
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)
    
    for _ in tf.range(tf.constant(mc_samples,dtype=tf.int32)):
        y= mc_step_milstein_default_l0(S0,const,n_points,mc_samples,num_timestep,y)

    return S0, y


@tf.function(jit_compile=True)
def mc_step_milstein_default_l0(init_S0,const,n_points,mc_samples,num_timestep,y):
    #print("Function mc_step_milstein_default_l0: tracing!")
    
    # Initialize Variables
    dt = const.T/num_timestep
    ST = init_S0

    # Iteratve over number of timesteps
    for i in range(num_timestep): 
        Si = tf.matmul(tf.sqrt(dt)*tf.random.normal(shape=(n_points, const.dim), dtype=const.DTYPE), const.C)
        ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

    y += fun_g_milstein_default(const,ST) / tf.cast(mc_samples,dtype=const.DTYPE)

    return y

@tf.function(jit_compile=True)
def gen_data_milstein_default_ln(const,n_points,mc_samples,num_timestep):
    #print("Function gen_data_milstein_default_ln: tracing!")

    # Initialize Variables
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)

    for _ in tf.range(tf.constant(mc_samples,dtype=tf.int32)):
        y= mc_step_milstein_default_ln(S0,const,n_points,mc_samples,num_timestep,y)
    return S0, y


@tf.function(jit_compile=True)
def mc_step_milstein_default_ln(init_S0,const,n_points,mc_samples,num_timestep,y):
    #print("Function mc_step_milstein_default_ln: tracing!")

    # Initialize Variables
    y_sample = [0.,0.]
    dt = const.T/num_timestep

    samplers = []
    samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, const.dim), dtype=const.DTYPE))
    samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,const.dim)),axis = 1))

    for k in range(2):
        ST = init_S0

        # Skip second iteration if Level 0
        dt =  const.T/samplers[k].shape[0]

        # Iteratve over number of timesteps
        for i in range(samplers[k].shape[0]):
            Si = tf.matmul(samplers[k][i,:,:], const.C)
            ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

        y_sample[k] = fun_g_milstein_default(const,ST) / tf.cast(mc_samples,dtype=const.DTYPE)

    # Calculate Difference for levels greater 0
    y += (y_sample[0] - y_sample[1])

    return y


#================================================================================
# Configuration 5: Control Variate Milstein
#================================================================================

@tf.function(jit_compile=True)
def fun_g_milstein_control_variate(const,x,init_S0):
    #print("Function fun_g_milstein_default: tracing!")
    basket_prices = tf.reduce_mean(x, axis=1)

    # Payoff and control variate
    F = tf.exp(-const.r*const.T) * tf.maximum(basket_prices - const.K,0.)
    C = tf.exp(-const.r*const.T) * basket_prices

    Fave = tf.reduce_mean(F,axis = 0)
    Cave = tf.reduce_mean(C,axis = 0)

    lam = tf.reduce_sum(tf.math.multiply((F-Fave),(C-Cave)),axis=0)/tf.reduce_sum(tf.math.pow((C-Cave),tf.constant(2,dtype=const.DTYPE)))
    price = tf.expand_dims(F - lam * (C - tf.reduce_mean(init_S0,axis=1)),axis=1)  
    return price

@tf.function(jit_compile=True)
def gen_data_milstein_control_variate_ln(const,n_points,mc_samples,num_timestep):
    #print("Function gen_data_milstein_default_ln: tracing!")

    # Initialize Variables
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)

    for _ in tf.range(tf.constant(mc_samples,dtype=tf.int32)):
        y= mc_step_milstein_control_variate_ln(S0,const,n_points,mc_samples,num_timestep,y)
    return S0, y


@tf.function(jit_compile=True)
def mc_step_milstein_control_variate_ln(init_S0,const,n_points,mc_samples,num_timestep,y):
    #print("Function mc_step_milstein_default_ln: tracing!")

    # Initialize Variables
    y_sample = [0.,0.]
    dt = const.T/num_timestep

    samplers = []
    samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, const.dim), dtype=const.DTYPE))
    samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,const.dim)),axis = 1))

    for k in range(2):
        ST = init_S0

        # Skip second iteration if Level 0
        dt =  const.T/samplers[k].shape[0]

        # Iteratve over number of timesteps
        for i in range(samplers[k].shape[0]):
            Si = tf.matmul(samplers[k][i,:,:], const.C)
            ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

        y_sample[k] = fun_g_milstein_control_variate(const,ST,init_S0) / tf.cast(mc_samples,dtype=const.DTYPE)

    # Calculate Difference for levels greater 0
    y += (y_sample[0] - y_sample[1])

    return y

@tf.function(jit_compile=True)
def gen_data_milstein_control_variate_l0(const,n_points,mc_samples,num_timestep):
    #print("Function gen_data_milstein_control_variate: tracing!")
    # Initialize Variables
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)

    for _ in tf.range(tf.constant(mc_samples,dtype=tf.int32)):
        y= mc_step_milstein_control_variate_l0(S0,const,n_points,mc_samples,num_timestep,y)

    return S0, y


@tf.function(jit_compile=True)
def mc_step_milstein_control_variate_l0(init_S0,const,n_points,mc_samples,num_timestep,y):
    ##print("Function mc_step_milstein_control_variate: tracing!")
    # Initialize Variables
    dt = const.T/num_timestep
    ST = init_S0

    # Iteratve over number of timesteps
    for _ in range(num_timestep):
        Si = tf.matmul(tf.sqrt(dt)*tf.random.normal(shape=(n_points, const.dim), dtype=const.DTYPE), const.C)
        ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si + 0.5 * tf.matmul(ST, const.sigma**2) * (Si**2 - dt)

    basket_prices = tf.reduce_mean(ST, axis=1)

    # Payoff and control variate
    y = y + (fun_g_milstein_control_variate(const,ST,init_S0) / tf.cast(mc_samples,dtype=const.DTYPE))
    return y


#================================================================================
# Configuration 6: Default Euler Method
#================================================================================

@tf.function(jit_compile=True)
def fun_g_euler_default(const,x):
        #print("Function fun_g_euler_default: tracing!")  
        return tf.exp(-const.r*const.T) * tf.maximum(tf.reduce_sum(x/const.dim, axis=1, keepdims=True) - const.K, 0.)

@tf.function(jit_compile=True)
def gen_data_euler_default_l0(const,n_points,mc_samples,num_timestep):
    #print("Function gen_data_euler_default_l0: tracing!")
    # Initialize Variables
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)

    for _ in tf.range(tf.constant(mc_samples,dtype=tf.int32)):
        y= mc_step_euler_default_l0(S0,const,n_points,mc_samples,num_timestep,y)

    return S0, y


@tf.function(jit_compile=True)
def mc_step_euler_default_l0(init_S0,const,n_points,mc_samples,num_timestep,y):
    #print("Function mc_step_euler_default_l0: tracing!")

    # Initialize Variables
    dt = const.T/num_timestep
    ST = init_S0

    # Iteratve over number of timesteps
    for i in range(num_timestep):
        Si = tf.matmul(tf.sqrt(dt)*tf.random.normal(shape=(n_points, const.dim), dtype=const.DTYPE), const.C)
        ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si

    y += fun_g_euler_default(const,ST) / tf.cast(mc_samples,dtype=const.DTYPE)

    return y

@tf.function(jit_compile=True)
def gen_data_euler_default_ln(const,n_points,mc_samples,num_timestep):
    #print("Function gen_data_euler_default_ln: tracing!")

    # Initialize Variables
    y = tf.zeros(
            shape=(n_points,1),
            dtype=const.DTYPE
            )
    S0 = const.a + tf.random.uniform((n_points,const.dim), dtype=const.DTYPE) * (const.b-const.a)

    for _ in tf.range(tf.constant(mc_samples,dtype=tf.int32)):
        y= mc_step_euler_default_ln(S0,const,n_points,mc_samples,num_timestep,y)
    return S0, y


@tf.function(jit_compile=True)
def mc_step_euler_default_ln(init_S0,const,n_points,mc_samples,num_timestep,y):
    #print("Function mc_step_euler_default_ln: tracing!")

    # Initialize Variables
    y_sample = [0.,0.]
    dt = const.T/num_timestep

    samplers = []
    samplers.append(tf.sqrt(dt)*tf.random.normal(shape=(num_timestep,n_points, const.dim), dtype=const.DTYPE))
    samplers.append(tf.reduce_sum(tf.reshape(samplers[0],shape = (int(num_timestep/2),2,n_points,const.dim)),axis = 1))

    for k in range(2):
        ST = init_S0

        # Skip second iteration if Level 0
        dt =  const.T/samplers[k].shape[0]

        # Iteratve over number of timesteps
        for i in range(samplers[k].shape[0]):
            Si = tf.matmul(samplers[k][i,:,:], const.C)
            ST = ST + ST * const.r * dt + tf.matmul(ST, const.sigma) * Si

        y_sample[k] = fun_g_euler_default(const,ST) / tf.cast(mc_samples,dtype=const.DTYPE)

    # Calculate Difference for levels greater 0
    y += (y_sample[0] - y_sample[1])

    return y
