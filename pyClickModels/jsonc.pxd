cdef extern from "/usr/include/json-c/json.h":
    struct json_object:
        pass

    ctypedef bint json_bool
    json_object *json_tokener_parse(const char *str)
    json_bool json_object_object_get_ex(const json_object *obj, const char *key, json_object **value)
    const char *json_object_get_string(json_object *jso)

    # ctypedef unsigned long (*lh_hash_fn)(const void *k)
    # ctypedef int (*lh_equal_fn)(const void *k1, const void *k2)

#     struct lh_table:
        # int size
        # lh_hash_fn *hash_fn
        # lh_equal_fn *equal_fn
#     ctypedef lh_table lh_table

    struct lh_entry:
        void *k
        void *v
        lh_entry *next

    struct lh_table:
        int size
        lh_entry *head

    lh_table *json_object_get_object(const json_object *jso)

    void *lh_entry_k(lh_entry *entry)
    size_t json_object_array_length(const json_object *obj)
    json_object *json_object_array_get_idx(const json_object *jso, size_t idx)
    int json_object_get_int(const json_object *obj)
