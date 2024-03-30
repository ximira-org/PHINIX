#!/usr/bin/env python3
import os

# Topic channel for all Module outputs (for juggler input)
TOPIC_MODULE_MESSAGES = "/phinix_ui_message_juggler/module_messages"

# Topic channel for Juggler outputs (if not sorting by periphery type)
TOPIC_JUGGLER_MESSAGES = "/phinix_ui_message_juggler/juggler_messages"

# Topic channels for each periphery type (for juggler output)
TOPIC_HAPTICS = "/phinix_ui_message_juggler/haptics_events"
TOPIC_VOICE = "/phinix_ui_message_juggler/voice_events"
TOPIC_SOUND = "/phinix_ui_message_juggler/sound_events"

