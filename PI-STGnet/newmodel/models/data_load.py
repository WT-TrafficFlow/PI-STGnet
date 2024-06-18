# -- coding: utf-8 --
from models.inits import *


def seq2instance(data, P, Q, low_index=0, high_index=100, granularity=5, sites=108, type='train'):
    '''
    :param data:
    :param P:
    :param Q:
    :param low_index:
    :param high_index:
    :param granularity:
    :param sites:
    :param type:
    :return: (N, sites, P) (N, sites, P+Q) (N, sites, P+Q) (N, sites, P+Q) (N, sites, P+Q) (N, 207, 24) (N, sites, P+Q)
    '''
    POI, X, DoW, D, H, M, L, XAll = [], [], [], [], [], [], [], []
    total_week_len = 60 // granularity * 24 * 7

    while low_index + P + Q < high_index:
        label1 = data[low_index * sites: (low_index + P + Q) * sites, -5:-4]
        label2 = data[low_index * sites: (low_index + P + Q) * sites, -6:-5]
        label = np.concatenate([label1, label2], axis=-1)[np.newaxis, :]
        label = np.concatenate([label[:, i * sites: (i + 1) * sites, :] for i in range(Q + P)], axis=0)[np.newaxis, :]  # l *site *2
        poi1 = data[low_index * sites: (low_index + P) * sites, 7:8]
        poi2 = data[low_index * sites: (low_index + P) * sites, -4:]
        poi = np.concatenate([poi1, poi2], axis=-1)[np.newaxis, :]
        poi = np.concatenate([poi[:, i * sites: (i + 1) * sites, :] for i in range(P)], axis=0)[np.newaxis, :]
        POI.append(poi)
        date_raw = data[low_index * sites: (low_index + P + Q) * sites, 0]
        date = [s.split(' ')[0] for s in date_raw]
        X.append(label[:, 0:P, :, :])
        DoW.append(np.reshape(
            [datetime.date(int(char.split('-')[0]), int(char.split('-')[1]),
                           int(char.split('-')[2])).weekday() for char in date], [1, P + Q, sites]))
        D.append(np.reshape([int(char.split('-')[2]) for char in date], [1, P + Q, sites]))
        H.append(np.reshape(data[low_index * sites: (low_index + P + Q) * sites, 4], [1, P + Q, sites]))
        # hours_to_minutes = data[low_index * sites: (low_index + P + Q) * sites, 5] // (60 / granularity)
        minutes_index_of_day = data[low_index * sites: (low_index + P + Q) * sites, 5] #粒度是 60
        M.append(np.reshape(minutes_index_of_day // granularity, [1, P + Q, sites]))
        L.append(label)
        last_week_apeed = np.expand_dims(np.reshape(data[(low_index - total_week_len) * sites: (low_index - total_week_len + P + Q) * sites, -5],[1, P + Q, sites]), axis=-1)
        last_week_flow = np.expand_dims(np.reshape(data[(low_index - total_week_len) * sites: (low_index - total_week_len + P + Q) * sites, -4],[1, P + Q, sites]), axis=-1)

        XAll.append(np.concatenate([last_week_apeed, last_week_flow], axis=-1)) # 上周同期数据

        if type == 'train':
            low_index += 1
        else:
            low_index += 1

    return np.concatenate(X, axis=0), \
        np.concatenate(DoW, axis=0), \
        np.concatenate(D, axis=0), \
        np.concatenate(H, axis=0), \
        np.concatenate(M, axis=0), \
        np.concatenate(L, axis=0), \
        np.concatenate(XAll, axis=0), \
        np.concatenate(POI, axis=0)


def loadData(args):
    # Traffic
    df = pd.read_csv(args.file_train_s)
    Traffic = df.values
    # train/val/test
    total_samples = df.shape[0] // args.site_num

    train_low = 60 // args.granularity * 24 * 7
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.validate_ratio) * total_samples)

    # X, Y, day of week, day, hour, minute, label, all X
    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, trainPOI = seq2instance(Traffic,
                                                                               args.input_length,
                                                                               args.output_length,
                                                                               low_index=train_low,
                                                                               high_index=val_low,
                                                                               granularity=args.granularity,
                                                                               sites=args.site_num,
                                                                               type='train')
    print('training dataset has been loaded!')
    valX, valDoW, valD, valH, valM, valL, valXAll, valPOI = seq2instance(Traffic,
                                                                 args.input_length,
                                                                 args.output_length,
                                                                 low_index=val_low,
                                                                 high_index=test_low,
                                                                 granularity=args.granularity,
                                                                 sites=args.site_num,
                                                                 type='validation')
    print('validation dataset has been loaded!')
    testX, testDoW, testD, testH, testM, testL, testXAll, testPOI = seq2instance(Traffic,
                                                                        args.input_length,
                                                                        args.output_length,
                                                                        low_index=test_low,
                                                                        high_index=total_samples,
                                                                        granularity=args.granularity,
                                                                        sites=args.site_num,
                                                                        type='test')
    print('testing dataset has been loaded!')
    # normalization
    mean = np.mean(trainX.reshape(-1, 2),  axis=0)
    std1 = np.std(trainX.reshape(-1, 2)[:,0])
    std2 = np.std(trainX.reshape(-1, 2)[:, 1])
    std = np.array([std1, std2])
    trainX, trainXAll = (trainX - mean) / std, (trainXAll - mean) / std
    valX, valXAll = (valX - mean) / std, (valXAll - mean) / std
    testX, testXAll = (testX - mean) / std, (testXAll - mean) / std

    return (trainX, trainDoW, trainM, trainL, trainXAll, trainPOI,
            valX, valDoW, valM, valL, valXAll, valPOI,
            testX, testDoW, testM, testL, testXAll, testPOI,
            mean, std)

# trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, mean, std = loadData(para)
#
# print(trainX.shape, valX.shape, testX.shape)
