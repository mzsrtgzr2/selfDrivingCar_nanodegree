import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helpers import preprocess_pipeline, build_classifier_nvidia, get_batches
from sklearn.model_selection import train_test_split
from keras.utils import vis_utils as vizu


if __name__ == '__main__':
    # Get the image names and measurements
    data_path = 'data'
    print ('Loading data in "{}"...'.format(data_path))
    features_filenames, targets = preprocess_pipeline(data_path)
    data = list(zip(features_filenames, targets))
    train_data, val_data = train_test_split(data, test_size=0.15)
    print ('Got total of {} images'.format(len(data)))

    
    # build the model and train
    print ('Building the model and training...')
    model = build_classifier_nvidia()
    batch_size = 32
    nb_epoch = 3
    # Compiling and training the model
    model.compile(loss='mse', optimizer='adam')

    # Open the file
    with open('modelsummary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    vizu.plot_model(model, "ff.png", show_layer_names=False, show_shapes=True)

    history_object = model.fit_generator(
        get_batches(train_data, batch_size=batch_size),
        samples_per_epoch=len(train_data)*2, # because we flip the images
        validation_data=get_batches(val_data, batch_size=batch_size),
        nb_val_samples=len(val_data)*2, # because we flip the images
        nb_epoch=nb_epoch,
        verbose=1)

    model_output = 'model.h5'
    print ('Done! saving to "{}"'.format(model_output))
    model.save(model_output)

    graph_file = 'train_history.png'
    fig = plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model mean squared error loss')
    plt.ylabel('Mean squared error loss')
    plt.xlabel('Epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    fig.savefig(graph_file, dpi=fig.dpi)
    print ('Training graph in "{}"'.format(graph_file))

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])
