from app.ml import train_models

if __name__ == '__main__':
    metrics = train_models()
    print('Training completed.')
    print('Best model:', metrics['best_model'])
    for name, result in metrics['results'].items():
        print(name, result)
