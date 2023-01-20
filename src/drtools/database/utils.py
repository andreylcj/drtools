""" 
This module was created to archive useful 
functions which can be used 
in drtools.data_science.database 
package.

"""


from typing import List


def get_queries_from_sql_file(
    filepath: str,
    split: str=';'
) -> List[str]:
    """Get SQL queries from a SQL file.

    Parameters
    ----------
    filepath : str
        Path to SQL file.

    Returns
    -------
    List[str]
        List of queries found in SQL file.
    """
    # Open and read the file as a single buffer
    fd = open(filepath, 'r')
    sql_file = fd.read()
    fd.close()
    if split is not None:
        # all SQL commands (split on ';')
        sql_commands = sql_file.split(';')
    else:
        sql_commands = sql_file.split(';')
    return sql_commands