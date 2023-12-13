def flatten_dict(input_dict: dict, prefix: Optional[str] = None) -> dict:
    """flatten_dict gathers sub^n keys of nested dicts and flatten into a 1 layer dict

    Parameters
    ----------
    input_dict : dict
        the dict to be flattened
    prefix : Optional[str], optional
        only used during recursion, by default None

    Returns
    -------
    dict
        the flattened dict
    """
    out_dict = {}
    if not isinstance(input_dict, dict):
        if prefix is None:
            raise ValueError("prefix cannot be None with a scalar value")
        else:
            return {prefix: input_dict}
    for key, value in input_dict.items():
        if prefix is not None:
            key = prefix + "_" + key
        if isinstance(value, dict):
            out_dict.update(flatten_dict(value, prefix=key))
        elif isinstance(value, list):
            for i, subvalue in enumerate(value):
                out_dict.update(flatten_dict(subvalue, prefix=key + "_" + str(i)))
        else:
            out_dict[key] = value
    return out_dict

def sha1(*args, size: Union[None, int] = 8) -> str:
    sha1_creator = hashlib.sha1()
    for item in args:
        sha1_creator.update(str(item).encode())
    return sha1_creator.digest().hex()[:size]

def slugify(text, sep="_", allow: Union[list[str], None] = None):
    text = unidecode.unidecode(text).lower()
    pattern = "\w"
    if allow:
        pattern = "|".join(["\w"] + allow)
    return sep.join(re.findall(f"[{pattern}]+", text))


def slugify_for_mlflow(text: str, sep: str = "_") -> str:
    return slugify(text, sep=sep, allow=["_", "-", ".", " ", "/"])