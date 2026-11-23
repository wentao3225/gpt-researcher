import re

def sanitize_filename(filename: str) -> str:
    """
    对给定文件名做净化，将 Windows 文件路径中无效的字符替换为下划线（'_'）。

    此函数通过移除或替换 Windows 文件路径中不允许的字符，
    确保文件名兼容所有操作系统。具体替换以下字符：
    < > : " / \\ | ? *

    参数：
    filename (str): 要净化的原始文件名。

    返回：
    str: 净化后的文件名，无效字符被替换为下划线。
    
    Examples:
    >>> sanitize_filename('invalid:file/name*example?.txt')
    'invalid_file_name_example_.txt'
    
    >>> sanitize_filename('valid_filename.txt')
    'valid_filename.txt'
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
