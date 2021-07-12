from .utils import *
from sklearn.model_selection import train_test_split
import time

time_str = time.strftime("%Y%m%d-%H%M%S")


def data_upload(state):
    st.title('Training Mode 1: v1m')
    with st.form('v1m_upload'):
        st.header('Data Upload')
        table_upload = st.file_uploader('Upload DGE table:', ['zip'],
                                        help='gene by cell pandas table, as the input of the NN model, in pickle')
        labels_upload = st.file_uploader('Upload cell type labels:', ['txt'],
                                         help='list of type labels for each cell, as the output of '
                                              'the NN model, delimited by line feed')
        data_submitted = st.form_submit_button('Submit data')
        if data_submitted:
            state.v1m_table = pd.read_pickle(table_upload, compression='zip')
            state.v1m_labels = pd.read_table(labels_upload, header=None, index_col=None)
            state.v1m_data_submitted = True


def training_options(state):
    with st.form('v1m_options'):
        st.header('Training Options')
        test_size = st.slider('The ratio of the test set size:', 0.0, 1.0,
                              state.v1m_test_size if state.v1m_test_size is not None else 0.1)
        if_rand_state = st.checkbox('Use a specific random seed for test set splitting',
                                    state.v1m_if_rand_state if state.v1m_if_rand_state is not None else True)
        rand_state_num = st.number_input('Input the random seed for test set splitting:',
                                         value=state.v1m_rand_state_num if state.v1m_rand_state_num is not None else 76)
        epoch = st.number_input('Input the number of training epochs:', 1,
                                value=state.v1m_epoch if state.v1m_epoch is not None else 10)
        batch_size = st.number_input('Input the training batch size:', 1,
                                     value=state.v1m_batch_size if state.v1m_batch_size is not None else 1024)
        validation_split = st.number_input('Input the split rate for validation set:', 0.0, 1.0,
                                           state.v1m_validation_split if
                                           state.v1m_validation_split is not None else 0.2)
        shuffle = st.checkbox('Shuffle the input data', state.v1m_shuffle if state.v1m_shuffle is not None else True)
        opt_submitted = st.form_submit_button('Submit options')
        if opt_submitted:
            state.v1m_test_size = test_size
            state.v1m_if_rand_state = if_rand_state
            state.v1m_rand_state_num = rand_state_num
            state.v1m_epoch = epoch
            state.v1m_batch_size = batch_size
            state.v1m_validation_split = validation_split
            state.v1m_shuffle = shuffle
            state.v1m_opt_submitted = True


def page(state):
    data_upload(state)
    training_options(state)

    if state.v1m_data_submitted and state.v1m_opt_submitted and st.button('Run'):
        with st.spinner('Data processing...'):
            cat_input, mapping = class_encode(state.v1m_labels.values)
            X_train, X_test, Y_train, Y_test = train_test_split(state.v1m_table.values, cat_input)
            cell_type_weights = calc_class_weight(np.argmax(Y_train, axis=1))
            n_cells, n_genes = X_train.shape
            n_types = len(cell_type_weights)
            nn = build_model(n_genes, n_types)
        st.header('Training Board')
        emp1 = st.empty()
        cont = st.beta_container()
        with st.spinner('Training...'):
            history = nn.fit(X_train, Y_train, epochs=state.v1m_epoch, batch_size=state.v1m_batch_size,
                             validation_split=state.v1m_validation_split, shuffle=state.v1m_shuffle,
                             class_weight=cell_type_weights,
                             callbacks=[CustomCallback(emp1, cont, np.floor(n_cells / state.v1m_batch_size *
                                                                                    (1 - state.v1m_validation_split)))])
        # Result
        
        st.header('Results')
        st.pyplot(plot_acc(history.history))
        st.pyplot(plot_loss(history.history))
        
        # Download
        st.header('Download')
        st.markdown(downloader_buffer("trained_model_{}.h5".format(time_str), model2buffer(nn)), unsafe_allow_html=True)
        st.markdown(downloader_gene_list("gene_list_{}.txt".format(time_str), state.v1m_table.index),
                    unsafe_allow_html=True)
        st.markdown(downloader(mapping.to_csv(index=False), "id2celltype_{}.csv".format(time_str)),
                    unsafe_allow_html=True)
        

