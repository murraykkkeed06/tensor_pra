import tensorflow as tf

# 宣告常數A&B，後面的name參數，是要繪製tensorboard時所使用的名稱。
# 若沒有指定，或是重複名稱，則tensorboard會自動修改。
#A = tf.constant(50, name='const_A')
#B = tf.constant(100, name='const_B')
#width = tf.placeholder("int32",name="width")
#height = tf.placeholder("int32",name="height")
#sum = tf.multiply(width,height,name="sum")
w = tf.Variable(tf.random_normal([1,3]),dtype=tf.float32,name="W")
x = tf.placeholder("float32",[3,2],name="x")
b = tf.Variable(tf.random_normal([1,2]),dtype=tf.float32,name="b")
wx_b_sum = tf.matmul(w,x)+b
relu = tf.nn.sigmoid(wx_b_sum)
with tf.Session() as sess:
    # 就是這邊！
    # 使用 "with tf.name_scope('Run'):" 這句話可以畫出Run這個步驟。
    init = tf.global_variables_initializer()
    sess.run(init)
    with tf.name_scope('Run'):
        Y = sess.run(relu,feed_dict={x:[[1.,1.],[1.,1.],[1.,1.]]})
    print(Y)
    
    # 畫好步驟之後，要使用"tf.summary.FileWriter"把檔案寫到目標資料夾，
    # 第二個參數表示要把整個圖層放到graph參數內，這樣才能用tensorboard畫出來。
    train_writer = tf.summary.FileWriter('log/area',sess.graph)
    train_writer.close()
