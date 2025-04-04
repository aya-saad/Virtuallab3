"""
Updated models for FMU simulation with Graph LLM integration
"""
import datetime
from django.db import models
from django.utils import timezone
from django.contrib import admin
from django import forms


class FMUForm(forms.Form):
    """
    Form for submitting FMU simulation questions
    """
    exp_desc = forms.CharField(
        required=True,
        widget=forms.widgets.Textarea(
            attrs={
                "placeholder": "Describe your experiment or ask a question...",
                "class": "textarea",
                "rows": 4,
            }
        ),
        label="Ask a question",
        max_length=512
    )
    exp_title = forms.CharField(widget=forms.HiddenInput(), required=False)
    exp_num = forms.CharField(widget=forms.HiddenInput(), required=False)

    # New fields for Graph LLM integration
    document_selector = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        label="Select documents to search (optional)",
        choices=[]  # Will be populated dynamically in the view
    )

    chat_mode = forms.ChoiceField(
        required=False,
        widget=forms.RadioSelect,
        label="Search mode",
        choices=[
            ('vector', 'Text Search'),
            ('graph', 'Graph Search'),
            ('entity', 'Entity Search'),
        ],
        initial='graph'
    )


class FMUModel(models.Model):
    """
    Model for storing FMU simulations
    """
    exp_num = models.IntegerField()
    exp_title = models.TextField(max_length=100)
    exp_desc = models.TextField(max_length=512)

    def __str__(self):
        return self.exp_title


class ChatSession(models.Model):
    """
    Model for storing chat sessions
    """
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Chat Session {self.session_id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class ChatMessage(models.Model):
    """
    Model for storing chat messages
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # For tracking document sources
    sources = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role.capitalize()} message at {self.created_at.strftime('%H:%M:%S')}"