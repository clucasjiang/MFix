import unittest
from pathlib import Path
from unittest.mock import patch

from src.credentials import CredentialError, load_settings, migrate_env, save_settings
from test_loop import WorkspaceTemporaryDirectory


class MemoryStore:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values.get(name, "")

    def set(self, name, secret):
        self.values[name] = secret


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.temp = WorkspaceTemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / ".env"
        self.store = MemoryStore()

    def test_migration_and_restart_preserve_keys_and_only_nonsecret_file_settings(self):
        self.path.write_text("# Preserve this comment\nCHAT_GPT_KEY='dummy-openai'\n"
                             "XAI_API_KEY=dummy-grok\nELEVENLABS_API_KEY=dummy-voice\n"
                             "OPENAI_MODEL=test-model\nELEVENLABS_VOICE_ID=voice-id\n", encoding="utf-8")
        initial = load_settings(self.path, store=self.store, environ={})
        self.assertEqual(initial["OPENAI_API_KEY"], "dummy-openai")
        self.assertEqual(initial["ELEVENLABS_API_KEY"], "dummy-voice")
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("dummy", text)
        self.assertIn("# Preserve this comment", text)
        self.assertIn("OPENAI_MODEL=test-model", text)
        self.assertIn("ELEVENLABS_VOICE_ID=voice-id", text)
        self.assertEqual(initial, load_settings(self.path, store=self.store, environ={}))

    def test_write_or_verification_failure_never_removes_legacy_file(self):
        original = "OPENAI_API_KEY=dummy-key\nXAI_MODEL=model\n"
        self.path.write_text(original, encoding="utf-8")
        with patch.object(self.store, "set", side_effect=CredentialError("storage unavailable")):
            with self.assertRaises(CredentialError):
                migrate_env(self.path, self.store)
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)
        with patch.object(self.store, "get", return_value=""):
            with self.assertRaises(CredentialError):
                migrate_env(self.path, self.store)
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_conflicting_legacy_key_is_not_discarded_or_allowed_to_replace_saved_key(self):
        self.path.write_text("OPENAI_API_KEY=dummy-old\n", encoding="utf-8")
        self.store.values["OPENAI_API_KEY"] = "dummy-new"
        with self.assertRaises(CredentialError):
            migrate_env(self.path, self.store)
        self.assertEqual(self.store.get("OPENAI_API_KEY"), "dummy-new")
        self.assertIn("dummy-old", self.path.read_text(encoding="utf-8"))

    def test_differing_alias_is_not_discarded(self):
        text = "OPENAI_API_KEY=dummy-primary\nCHAT_GPT_KEY=dummy-alias\n"
        self.path.write_text(text, encoding="utf-8")
        with self.assertRaises(CredentialError):
            migrate_env(self.path, self.store)
        self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_settings_save_updates_secure_keys_and_blank_keeps_existing(self):
        save_settings(self.path, {"OPENAI_API_KEY": "dummy-model", "ELEVENLABS_API_KEY": "dummy-voice",
                                 "MODEL_PROVIDER": "openai"}, store=self.store, environ={})
        save_settings(self.path, {"OPENAI_API_KEY": "", "OPENAI_SEARCH": "off"}, store=self.store, environ={})
        values = load_settings(self.path, store=self.store, environ={})
        self.assertEqual(values["OPENAI_API_KEY"], "dummy-model")
        self.assertEqual(values["ELEVENLABS_API_KEY"], "dummy-voice")
        self.assertEqual(values["OPENAI_SEARCH"], "off")
        self.assertNotIn("API_KEY", self.path.read_text(encoding="utf-8"))

    def test_process_override_and_alias_do_not_get_persisted(self):
        self.store.values["OPENAI_API_KEY"] = "dummy-saved"
        values = load_settings(self.path, store=self.store, environ={"CHAT_GPT_KEY": "dummy-process"})
        self.assertEqual(values["OPENAI_API_KEY"], "dummy-process")
        self.assertEqual(self.store.get("OPENAI_API_KEY"), "dummy-saved")
        self.assertFalse(self.path.exists())

    def test_atomic_file_failure_leaves_source_and_verified_vault_keys_available(self):
        original = "OPENAI_API_KEY=dummy-key\n"
        self.path.write_text(original, encoding="utf-8")
        with patch("dotenv.main.os.replace", side_effect=PermissionError("file locked")):
            with self.assertRaises(PermissionError):
                migrate_env(self.path, self.store)
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)
        self.assertEqual(self.store.get("OPENAI_API_KEY"), "dummy-key")

    def test_export_duplicate_and_multiline_key_entries_are_all_removed(self):
        self.path.write_text("export OPENAI_API_KEY='dummy-first'\n"
                             "OPENAI_API_KEY='dummy-last'\nELEVENLABS_API_KEY='dummy\nvoice'\n"
                             "# Keep\nOPENAI_MODEL=test\n", encoding="utf-8")
        migrate_env(self.path, self.store)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "# Keep\nOPENAI_MODEL=test\n")
        self.assertEqual(self.store.get("OPENAI_API_KEY"), "dummy-last")
