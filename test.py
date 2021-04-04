from utility import restore_patch, plot_result

def test_model(model, X, Y):
    #e1 = model.evaluate(X[700:800], Y[700:800], True)
    test_loss = model.evaluate(X[800:], Y[800:], False)
    print('Test Loss {:.4f}'.format(test_loss))

    y1 = model.predict(X[50], 10)
    y2 = model.predict(X[915], 10)
    y3 = model.predict(X[936], 10)
    y4 = model.predict(X[956], 10)

    plot_result(
        restore_patch(X[50].numpy(), (2, 2)),
        restore_patch(Y[50].numpy(), (2, 2)),
        restore_patch(y1, (2, 2))
    )
    
    plot_result(
        restore_patch(X[915].numpy(), (2, 2)),
        restore_patch(Y[915].numpy(), (2, 2)),
        restore_patch(y2, (2, 2))
    )
    
    plot_result(
        restore_patch(X[936].numpy(), (2, 2)),
        restore_patch(Y[936].numpy(), (2, 2)),
        restore_patch(y3, (2, 2))
    )
    
    plot_result(
        restore_patch(X[956].numpy(), (2, 2)),
        restore_patch(Y[956].numpy(), (2, 2)),
        restore_patch(y4, (2, 2))
    )
