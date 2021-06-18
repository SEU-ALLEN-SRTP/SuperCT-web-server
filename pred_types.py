import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import base64
import time
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP


timestr = time.strftime("%Y%m%d-%H%M%S")


@st.cache(allow_output_mutation=True)
def load_genes(genes_file, species):
    genes_used = pd.read_csv(genes_file, usecols=[0 if species == 'human' else 1], header=None).values.tolist()
    for i, g in enumerate(genes_used):
        genes_used[i] = g[0]
    return genes_used


@st.cache(allow_output_mutation=True)
def load_data(file, genes_used):
    # file can be an IO buffer or path
    data = pd.read_csv(file, header=0, index_col=0)
    data = data[~data.index.duplicated(keep='first')]
    genes_valid = np.intersect1d(data.index, genes_used)
    if len(genes_valid) == len(genes_used):
        data = data.loc[genes_used, :]
    elif len(genes_valid) < len(genes_used):
        genes_not_found = np.setdiff1d(genes_used, genes_valid)
        nonezero_df = pd.DataFrame(np.zeros((len(genes_not_found), data.shape[1])).astype('int64'))
        nonezero_df.columns = data.columns
        data = pd.concat([data, nonezero_df])
        data = data.loc[genes_used, :]
        print('Warning:', genes_not_found, ''' not found in your expression file,please check them. 
        But the prediction will go on by setting these genes with zero count.''')
    return data


@st.cache(allow_output_mutation=True)
def load_cell_types(types):
    cell_types = {}
    all_types = pd.read_csv(types)
    for i, each in enumerate(all_types['celltype']):
        cell_types[i] = each
    return cell_types


def predict_type(model, data, cell_types):
    data = data > 0
    data = data.astype(np.uint8)
    result = model.predict(data.values.T)
    result_df = pd.DataFrame(result)
    result_df.columns = list(cell_types.values())
    cell_type_pred = [cell_types[i] for i in list(np.argmax(result, axis=1))]
    result_df["pred_type"] = cell_type_pred
    return result_df


def downloader(text, fname):
    b64 = base64.b64encode(text.encode()).decode()
    label = fname.replace("_", r"\_")
    href = f'<a href="data:file/csv;base64,{b64}" download="{fname}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


def plot_histogram(res):
    d = {}
    res_list = list(res['pred_type'])
    for i in res_list:
        d.update({i: res_list.count(i)})
    plt.bar(d.keys(), d.values())
    plt.title('Frequency of predicted cell types')
    plt.xticks(rotation=90)
    st.pyplot(plt, bbox_inches='tight')


@st.cache(allow_output_mutation=True)
def run_pca(df, n_comp=50):
    pca = PCA(n_components=n_comp)
    dr = pca.fit_transform(df.transpose())
    return dr


@st.cache(allow_output_mutation=True)
def run_tsne(dr, n_comp=2):
    tsne = TSNE(n_components=n_comp)
    projections = tsne.fit_transform(dr)
    return projections


@st.cache(allow_output_mutation=True)
def run_umap(dr, n_comp=2):
    umap = UMAP(n_components=n_comp)
    projections = umap.fit_transform(dr)
    return projections


@st.cache(allow_output_mutation=True)
def plot_umap(df, res, as3d):
    dr = run_pca(df)
    projections = run_umap(dr, 3 if as3d else 2)
    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=res.pred_type, labels={'color': 'pred_type'}
    ) if as3d else px.scatter(
        projections, x=0, y=1,
        color=res.pred_type, labels={'color': 'pred_type'}
    )
    fig.update_traces(marker_size=5)
    return fig


@st.cache(allow_output_mutation=True)
def plot_tsne(df, res):
    dr = run_pca(df)
    projections = run_tsne(dr)
    fig = px.scatter(
        projections, x=0, y=1,
        color=res.pred_type, labels={'color': 'pred_type'}
    )
    fig.update_traces(marker_size=5)
    return fig


@st.cache(allow_output_mutation=True)
def load_model_cache(use_model):
    model = load_model('models/' + use_model)
    return model


def main():
    st.sidebar.title('SuperCT Web Server')

    st.title('Cell type mapping')
    st.text('Mapping your scRNA-seq data to cell types, with our neural network model.')

    # side
    st.sidebar.header('Data')
    eg = st.sidebar.checkbox('Example: human')
    upload = 'example/input_dge.human.csv' if eg else \
        st.sidebar.file_uploader('Upload your DGE table', ['csv'],
                                 help='differential gene expression table (csv), binary')
    spc = st.sidebar.selectbox('Select the species of your data', ['human', 'mouse']) if not eg else 'human'
    genes_used = load_genes('inthomgenes.csv', spc)

    # page
    st.header('Options')
    use_model = st.selectbox('Select a trained neural network model', ['v1_model.h5'])
    model = load_model_cache(use_model)
    cell_types = load_cell_types('models/' + st.selectbox('Select a label mapping', ['v1_id2type.csv']))

    plot_hist = st.checkbox('Plot Cell Type Histogram', True)
    dr_umap = st.checkbox('View mapping results in UMAP', True)
    dr_tsne = st.checkbox('View mapping results in tSNE', False)
    plot3d_umap = st.checkbox('Enable 3D UMAP', True)

    if upload is not None:
        data = load_data(upload, genes_used)
        if eg:
            st.sidebar.write('First 20 rows and 20 columns...')
            st.sidebar.dataframe(data.iloc[0:19, 0:19])
        if st.button('Run Prediction'):
            with st.spinner("Running..."):
                st.header('Results')
                result_df = predict_type(model, data, cell_types)
                result_out = pd.DataFrame({'cell_id': data.columns, 'pred_type': result_df['pred_type']})
                with st.beta_expander('Predictions (first 20 rows)', True):
                    st.dataframe(result_out.iloc[0:19])
                if plot_hist:
                    with st.beta_expander('Cell type histogram', True):
                        plot_histogram(result_out)
                if dr_umap:
                    with st.beta_expander('UMAP Plot', True):
                        st.plotly_chart(plot_umap(data, result_out, plot3d_umap), use_container_width=True)
                if dr_tsne:
                    with st.beta_expander('tSNE Plot', True):
                        st.plotly_chart(plot_tsne(data, result_out), use_container_width=True)
            st.subheader('Download')
            downloader(result_out.to_csv(index=False), "cell_type_prediction_{}.csv".format(timestr))


if __name__ == '__main__':
    main()
