"""Current-user Windows credentials; never fall back to plaintext writes."""
import ctypes
import os
from ctypes import wintypes

from dotenv import dotenv_values
from dotenv.main import rewrite
from dotenv.parser import parse_stream

KEY_NAMES = ("OPENAI_API_KEY", "XAI_API_KEY", "ELEVENLABS_API_KEY")
SECRET_NAMES = (*KEY_NAMES, "CHAT_GPT_KEY")


class CredentialError(RuntimeError):
    pass


class _Credential(ctypes.Structure):
    _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(wintypes.BYTE)), ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]


class WindowsCredentials:
    def __init__(self, namespace="MHacks.Helper"):
        if os.name != "nt":
            raise CredentialError("Secure key storage requires Windows.")
        self.namespace = namespace
        self.api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        pointer = ctypes.POINTER(_Credential)
        self.api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(pointer)]
        self.api.CredReadW.restype = wintypes.BOOL
        self.api.CredWriteW.argtypes = [pointer, wintypes.DWORD]
        self.api.CredWriteW.restype = wintypes.BOOL
        self.api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.api.CredDeleteW.restype = wintypes.BOOL
        self.api.CredFree.argtypes = [ctypes.c_void_p]
        self.api.CredFree.restype = None

    def _target(self, name):
        if name not in KEY_NAMES:
            raise CredentialError("Unsupported credential name.")
        return f"{self.namespace}/{name}"

    @staticmethod
    def _failure():
        return CredentialError(f"Windows could not access secure key storage (code {ctypes.get_last_error()}).")

    def get(self, name):
        pointer = ctypes.POINTER(_Credential)()
        if not self.api.CredReadW(self._target(name), 1, 0, ctypes.byref(pointer)):
            if ctypes.get_last_error() == 1168:  # ERROR_NOT_FOUND
                return ""
            raise self._failure()
        try:
            blob = ctypes.string_at(pointer.contents.CredentialBlob, pointer.contents.CredentialBlobSize)
            try:
                return blob.decode("utf-16-le")
            except UnicodeError:
                raise CredentialError("The saved key could not be read. Enter it again in Settings.") from None
        finally:
            self.api.CredFree(pointer)

    def set(self, name, secret):
        encoded = secret.encode("utf-16-le")
        if not encoded or len(encoded) > 2560:
            raise CredentialError("The API key is empty or too long to store.")
        buffer = (wintypes.BYTE * len(encoded)).from_buffer_copy(encoded)
        credential = _Credential(Type=1, TargetName=self._target(name),
                                 Comment="Helper API key", CredentialBlobSize=len(encoded),
                                 CredentialBlob=buffer, Persist=2, UserName="helper")
        try:
            if not self.api.CredWriteW(ctypes.byref(credential), 0):
                raise self._failure()
        finally:
            ctypes.memset(buffer, 0, len(encoded))

    def delete(self, name):
        if not self.api.CredDeleteW(self._target(name), 1, 0) and ctypes.get_last_error() != 1168:
            raise self._failure()


def store_verified(store, name, secret):
    store.set(name, secret)
    if store.get(name) != secret:
        raise CredentialError("Windows did not confirm the saved key. Please try again.")


def migrate_env(path, store):
    """Verify all nonblank keys before atomically removing plaintext entries."""
    if not path.exists():
        return
    values = dotenv_values(path, interpolate=False)
    if not any(name in values for name in SECRET_NAMES):
        return
    pending = {name: (values.get(name) or "").strip() for name in KEY_NAMES}
    pending["OPENAI_API_KEY"] = pending["OPENAI_API_KEY"] or (values.get("CHAT_GPT_KEY") or "").strip()
    alias = (values.get("CHAT_GPT_KEY") or "").strip()
    if alias and pending["OPENAI_API_KEY"] != alias:
        raise CredentialError("The two legacy OpenAI key entries differ. Keep the desired key in one entry before continuing.")
    for name, secret in pending.items():
        if secret:
            existing = store.get(name)
            # Preserve a newer key already saved by Settings. A conflicting legacy
            # value stays on disk until the user resolves it; never discard it.
            if existing and existing != secret:
                raise CredentialError("A file key differs from the securely saved key. Update or remove the legacy file key before continuing.")
            store_verified(store, name, secret)
    with rewrite(path, encoding="utf-8") as (source, destination):
        for binding in parse_stream(source):
            if binding.key not in SECRET_NAMES:
                destination.write(binding.original.string)


def load_settings(path, *, store=None, environ=None, migrate=True):
    store = store if store is not None else WindowsCredentials()
    if migrate:
        migrate_env(path, store)
    values = {name: value for name, value in dotenv_values(path).items() if name not in SECRET_NAMES}
    values.update({name: store.get(name) for name in KEY_NAMES})
    environment = dict(os.environ if environ is None else environ)
    # Preserve legacy alias support for an explicitly supplied process key.
    openai = (environment.get("OPENAI_API_KEY") or "").strip() or (environment.get("CHAT_GPT_KEY") or "").strip()
    values.update(environment)
    if openai:
        values["OPENAI_API_KEY"] = openai
    return values


def save_settings(path, changes, *, store=None, environ=None):
    from dotenv import set_key
    store = store if store is not None else WindowsCredentials()
    migrate_env(path, store)
    for name, value in changes.items():
        if name in SECRET_NAMES:
            if value.strip():
                store_verified(store, "OPENAI_API_KEY" if name == "CHAT_GPT_KEY" else name, value.strip())
        else:
            set_key(path, name, value)
    return load_settings(path, store=store, environ=environ, migrate=False)
