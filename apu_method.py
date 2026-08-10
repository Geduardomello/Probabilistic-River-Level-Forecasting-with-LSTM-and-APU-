
import numpy as np
import tensorflow as tf

def apu_uncertainty_lstm(
    model,
    x_seq,
    u_x_seq,
    SE,
    RE,
    u_y,
    k_coverage=2.0,
    rho=None
):
    """
    Calcula a incerteza de previsão de uma LSTM com base no método APU.
    Retorna:
      - y_pred
      - u_D, u_M, u_P e U
      - Matrizes M1, M2, M3, M4
    """

    x = np.asarray(x_seq, dtype=float).reshape(1, -1, 1)
    k = x.shape[1]
    with tf.GradientTape() as tape:
        xt = tf.convert_to_tensor(x, dtype=tf.float32)
        tape.watch(xt)
        y_pred = model(xt, training=False)
    grads = tape.gradient(y_pred, xt).numpy().reshape(-1)

    u = np.asarray(u_x_seq, dtype=float).reshape(-1)
    if u.size != k:
        raise ValueError(f"u_x_seq deve ter tamanho {k}.")

    M1 = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i, k):
            M1[i, j] = u[i] * u[j]

    M2 = np.zeros((k, k), dtype=float)
    for i in range(k):
        M2[i, i] = grads[i] ** 2
        for j in range(i + 1, k):
            M2[i, j] = 2.0 * grads[i] * grads[j]

    if rho is None:
        M3 = np.ones((k, k), dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)
        if rho.shape != (k, k):
            raise ValueError(f"rho deve ter forma ({k}, {k})")
        M3 = np.triu(rho, 0)

    M4 = M1 * M2 * M3
    uD2_texto = M4.sum() + float(u_y) ** 2
    u_D = np.sqrt(uD2_texto)

    u_M = float(RE)
    uP2 = uD2_texto + u_M ** 2
    u_P = np.sqrt(uP2)

    U = float(k_coverage) * u_P + float(SE)

    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            R[i, j] = M3[i, j]
            R[j, i] = M3[i, j]

    Cx = (u[:, None] * R * u[None, :])
    uD2_compacto = grads @ (Cx @ grads) + float(u_y) ** 2

    return {
        "y_pred": float(y_pred.numpy()[0, 0]),
        "u_D": float(u_D),
        "u_M": float(u_M),
        "u_P": float(u_P),
        "U": float(U),
        "grads": grads,
        "M1": M1,
        "M2": M2,
        "M3": M3,
        "M4": M4,
        "uD2_texto": float(uD2_texto),
        "uD2_compacto": float(uD2_compacto),
        "Cx": Cx
    }
