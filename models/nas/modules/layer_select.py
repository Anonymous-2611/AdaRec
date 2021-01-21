import random


def layer_select(num_student_layers, num_teacher_layers, method):
    # method:
    #  - random
    #  - left | right | left-jump | right-jump
    if method == "random":
        for i in range(num_student_layers):
            rng = random.randint(0, num_teacher_layers - 1)
            yield i, rng
        return
    if num_teacher_layers == num_student_layers:
        for i in range(num_student_layers):
            yield i, i
    elif num_teacher_layers > num_student_layers:
        # e.g. |S| = 4, |T| = 8
        if method == "left":
            # IDX 0 1 2 3
            # GET 0 1 2 3
            # BIG 0 1 2 3 4 5 6 7
            for i in range(num_student_layers):
                yield i, i
        elif method == "right":
            # IDX         0 1 2 3
            # GET         4 5 6 7
            # BIG 0 1 2 3 4 5 6 7
            for i in range(num_student_layers):
                yield i, num_teacher_layers - num_student_layers + i
        elif method == "left-jump":
            # IDX 0   1   2   3
            # GET 0   2   4   6
            # BIG 0 1 2 3 4 5 6 7
            if num_student_layers == 1:
                yield 0, 0
                return
            delta = (num_teacher_layers - 1) // (num_student_layers - 1)
            for i in range(num_student_layers):
                teacher_layer = i * delta
                yield i, teacher_layer
        elif method == "right-jump":
            # IDX   0   1   2   3
            # GET   1   3   5   7
            # BIG 0 1 2 3 4 5 6 7
            if num_student_layers == 1:
                yield 0, num_teacher_layers - 1
                return
            delta = (num_teacher_layers - 1) // (num_student_layers - 1)
            start = (num_teacher_layers - 1) % (num_student_layers - 1)
            for i in range(num_student_layers):
                teacher_layer = start + i * delta
                yield i, teacher_layer
        else:
            raise ValueError("Unexpected layer select method `{}`".format(method))
    else:  # teacher_layers < target_layers:
        # e.g. |S| = 7, |T| = 3
        if method == "left":
            # IDX 0 1 2 3 4 5 6
            # GET 0 0 0 0 0 1 2
            # BIG         0 1 2
            diff = num_student_layers - num_teacher_layers
            for i in range(diff):
                yield i, 0
            for i in range(num_teacher_layers):
                yield i + diff, i
        elif method == "right":
            # IDX 0 1 2 3 4 5 6
            # GET 0 1 2 2 2 2 2
            # BIG 0 1 2
            diff = num_student_layers - num_teacher_layers
            for i in range(num_teacher_layers):
                yield i, i
            for i in range(diff):
                yield i + num_teacher_layers, num_teacher_layers - 1
        elif method == "left-jump":
            # IDX 0 1 2 3 4 5 6
            # GET 0 0 1 1 2 2 2
            #     A > B > C > >
            # BIG 0   1   2
            # fill ABC and expand
            repeat = num_student_layers // num_teacher_layers
            cur_s = 0
            for teacher_layer in range(num_teacher_layers):
                for _ in range(repeat):
                    yield cur_s, teacher_layer
                    cur_s += 1
            for i in range(cur_s, num_student_layers):
                yield i, num_teacher_layers - 1
        elif method == "right-jump":
            # IDX 0 1 2 3 4 5 6
            # GET 0 0 0 1 1 2 2
            #     < < C < B < A
            # BIG     0   1   2
            # fill ABC and expand
            repeat = num_student_layers // num_teacher_layers
            cur_s = num_student_layers - repeat * num_teacher_layers
            for i in range(cur_s):
                yield i, 0
            for teacher_layer in range(num_teacher_layers):
                for _ in range(repeat):
                    yield cur_s, teacher_layer
                    cur_s += 1
        else:
            raise ValueError("Unexpected layer select method `{}`".format(method))


if __name__ == "__main__":
    for a, b in layer_select(1, 3, "right-jump"):
        print(a, b)
