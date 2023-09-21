embedding_dims = {
    'doc': 32,
    'domain': 10,
    'chn': 100,
    'cate': 8,
    'seg_title': 50,
    'os': 16,
    'uid': 16,
    'doc_em': 50,
    'zip_code': 16,
    'pid': 16,
	'level_score': 8,
    'on_boarding_days': 8,
    'county_fibs': 8,
    'city_id': 8,
	'dma_code': 8,
    'state_code': 8,
    'check': 8,
    'click': 8,
    'ctr': 8,
    'avg_level_score': 8,
}

num_embeddings = {
    'doc': 5000088,
    'domain': 30000,
    'cate': 800,
    'chn': 150000,
    'seg_title': 100000,
    'os': 3,
    'uid': 9999992,
    'zip_code': 200000,
    'pid': 100000,
    'on_boarding_days': 11,
    'county_fibs': 60000,
    'city_id': 30000,
	'dma_code': 300,
    'state_code': 60,
    'check': 21,
    'click': 21,
}

keep_from_batch = {
    'os',
    'uid',
    'label',
    'zip_code',
    'is_push'
}

combine_nonlocal_doc_emb_len = sum([
    embedding_dims['doc'],
    embedding_dims['domain'],
    embedding_dims['chn'],
    embedding_dims['cate'],
    embedding_dims['seg_title'],
    embedding_dims['doc_em'],
])

combine_doc_emb_len = combine_nonlocal_doc_emb_len + embedding_dims['pid']

feature_meta_size = {
	'level_score': 16,
    'ctr': 16,
    'avg_level_score': 16,
}

user_context_features = {
	'level_score',
    'on_boarding_days',
    'county_fibs',
    'state_code',
    'city_id',
	'dma_code',
}

cfb_features = {
    'check',
    'click',
    'ctr',
    'avg_level_score',
}

user_context_feature_emb_len = sum([ embedding_dims[k] for k in user_context_features ])
cfb_feature_emb_len = sum([ embedding_dims[k] for k in cfb_features ])