## Modified from https://github.com/luping-liu/PNDM/blob/f285e8e6da36049ea29e97b741fb71e531505ec8/runner/method.py#L20

def runge_kutta(x, t_list, model, alphas_cump, ets, clip_before=False):
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump, clip_before)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump, clip_before)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump, clip_before)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et, ets

def transfer(x, t, t_next, et, alphas_cump, clip_before=False):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    # x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * grad)
    # x_mod = c_alpha_prev.sqrt() * x0 + (1 - c_alpha_prev).sqrt() * grad

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    if clip_before:
        x_next = x_next.clip_(-1, 1)

    return x_next

def gen_order_1(img, t, t_next, model, alphas_cump, ets, clip_before=False): ## DDIM
    noise = model(img, t)
    ets.append(noise)
    img_next = transfer(img, t, t_next, noise, alphas_cump, clip_before)
    return img_next, ets

def gen_order_4(img, t, t_next, model, alphas_cump, ets, clip_before=False): ## F-PNDM
    t_list = [t, (t+t_next)/2, t_next]
    #print(t_list)
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise, ets = runge_kutta(img, t_list, model, alphas_cump, ets, clip_before)

    img_next = transfer(img, t, t_next, noise, alphas_cump, clip_before)
    return img_next, ets
