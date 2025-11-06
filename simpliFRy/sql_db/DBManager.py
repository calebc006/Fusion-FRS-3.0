import sqlite3
from contextlib import contextmanager
from typing import Generator, TypedDict

import numpy as np

from utils import log_info

DB_FP = "Embeddings.db"


class PersonRecord(TypedDict):
    name: str
    ave_embedding: np.ndarray


def adapt_array(arr: np.ndarray) -> bytes:
    """Convert NumPy array to binary (serialize)"""

    return arr.tobytes()


def convert_array(blob):
    """Convert binary back to NumPy array (deserialize)"""

    return np.frombuffer(blob, dtype=np.float32)


@contextmanager
def get_db() -> Generator[sqlite3.Connection, any, any]:
    """
    Provides the connection to SQLite database

    Returns
    - A generator yielding the connection to the SQLite database storing embeddings for FR
    """

    # Register adapters to handle numpy arrays as BLOBs
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("NP_ARRAY", convert_array)

    conn = sqlite3.connect(DB_FP, detect_types=sqlite3.PARSE_DECLTYPES)

    try:
        yield conn
    except sqlite3.Error as err:
        log_info(f"Error connecting to database: {err}")
    finally:
        conn.close()


def recreate_table(conn: sqlite3.Connection) -> None:
    """
    Delete all current records and create table storing embeddings if it does not exist

    Arguments
    - conn: connection to SQLite database 
    """

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Embeddings (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            embedding NP_ARRAY NOT NULL
       )
    """)
    cursor.execute("DELETE FROM Embeddings")
    conn.commit()
    log_info("DATABASE RESETTED")


def save_record(conn: sqlite3.Connection, name: str, ave_embedding: np.ndarray) -> None:
    """
    Adds a person to the SQLite database (name and embedding representation of face)

    Arguments
    - conn: connection to SQLite database
    - name: name of person
    - ave_embedding: embedding representation of face (average from all images of the person)
    """

    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO Embeddings (name, embedding) VALUES (?, ?)", (name, ave_embedding)
    )
    conn.commit()


def fetch_records(conn: sqlite3.Connection) -> list[PersonRecord]:
    """
    Fetch all embeddings from SQLite database and returns them in a python dictionary

    Arguments
    - conn: connection to SQLite database

    Returns
    - A list of python dictionaries; each dictionary stores the name and embedding representation of the face of an individual
    """

    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM Embeddings")
    results = cursor.fetchall()

    return [{"name": result[0], "ave_embedding": result[1]} for result in results]
