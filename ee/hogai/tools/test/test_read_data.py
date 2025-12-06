from uuid import uuid4

import pytest
from posthog.test.base import BaseTest
from unittest.mock import AsyncMock, MagicMock, patch

from posthog.schema import ArtifactSource, AssistantTrendsEventsNode, AssistantTrendsQuery, VisualizationArtifactContent

from ee.hogai.tool_errors import MaxToolRetryableError
from ee.hogai.tools.read_data import (
    READ_DATA_BILLING_PROMPT,
    READ_DATA_PROMPT,
    ReadDataAdminAccessToolArgs,
    ReadDataTool,
    ReadDataToolArgs,
)
from ee.hogai.utils.types import AssistantState
from ee.hogai.utils.types.base import NodePath


class TestReadDataTool(BaseTest):
    async def test_create_tool_class_with_billing_access(self):
        """Test that billing prompt is included when user has billing access."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=True)

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        # Should use the admin access args schema
        assert tool.args_schema == ReadDataAdminAccessToolArgs

        # Description should include billing prompt
        assert "billing_info" in tool.description
        assert "Billing information" in tool.description

    async def test_create_tool_class_without_billing_access(self):
        """Test that billing prompt is excluded when user lacks billing access."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=False)

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        # Should use the basic args schema
        assert tool.args_schema == ReadDataToolArgs

        # Description should NOT include billing prompt
        assert "billing_info" not in tool.description
        assert "Billing information" not in tool.description

        # Should still have base prompt content
        assert "data warehouse" in tool.description

    async def test_create_tool_class_formats_prompt_correctly(self):
        """Test that format_prompt_string correctly injects billing_prompt."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=True)

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        # The formatted description should contain both base prompt and billing prompt
        expected_description = READ_DATA_PROMPT.replace(
            "{{{billing_prompt}}}", READ_DATA_BILLING_PROMPT.strip()
        ).strip()
        assert tool.description == expected_description

    async def test_create_tool_class_without_context_manager(self):
        """Test that create_tool_class creates context manager if not provided."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))

        with patch("ee.hogai.tools.read_data.AssistantContextManager") as mock_context_class:
            mock_context = MagicMock()
            mock_context.check_user_has_billing_access = AsyncMock(return_value=False)
            mock_context_class.return_value = mock_context

            tool = await ReadDataTool.create_tool_class(
                team=team,
                user=user,
                state=state,
            )

            mock_context_class.assert_called_once()
            assert tool is not None

    async def test_read_insight_schema_only(self):
        """Test reading an insight without executing it returns the schema."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=False)

        mock_query = AssistantTrendsQuery(series=[AssistantTrendsEventsNode(name="$pageview")])
        mock_content = VisualizationArtifactContent(
            name="Test Insight",
            description="A test description",
            query=mock_query,
        )

        context_manager.artifacts = MagicMock()
        context_manager.artifacts.aget_insight_with_source = AsyncMock(
            return_value=(mock_content, ArtifactSource.INSIGHT)
        )

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        result, artifact = await tool._arun_impl({"kind": "insight", "insight_id": "abc123", "execute": False})

        assert "Test Insight" in result
        assert "abc123" in result
        assert "A test description" in result
        assert "TrendsQuery" in result
        assert artifact is None

    async def test_read_insight_with_execution(self):
        """Test reading an insight with execution returns results and artifact."""
        team = MagicMock()
        user = MagicMock()
        tool_call_id = "test_call_id"
        state = AssistantState(messages=[], root_tool_call_id=tool_call_id)
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=False)

        mock_query = AssistantTrendsQuery(series=[AssistantTrendsEventsNode(name="$pageview")])
        mock_content = VisualizationArtifactContent(
            name="Test Insight",
            description="A test description",
            query=mock_query,
        )

        context_manager.artifacts = MagicMock()
        context_manager.artifacts.aget_insight_with_source = AsyncMock(
            return_value=(mock_content, ArtifactSource.INSIGHT)
        )

        with patch(
            "ee.hogai.tools.read_data.execute_and_format_query", new=AsyncMock(return_value="Formatted results")
        ):
            tool = ReadDataTool(
                team=team,
                user=user,
                state=state,
                context_manager=context_manager,
                node_path=(NodePath(name="test_node", tool_call_id=tool_call_id, message_id="test"),),
            )

            result, artifact = await tool._arun_impl({"kind": "insight", "insight_id": "abc123", "execute": True})

            # When execute=True, returns empty string and artifact
            assert result == ""
            assert artifact is not None
            assert len(artifact.messages) == 2

            # First message is ArtifactRefMessage
            artifact_ref = artifact.messages[0]
            assert artifact_ref.artifact_id == "abc123"
            assert artifact_ref.source == ArtifactSource.INSIGHT

            # Second message is the tool call message with results
            tool_call_msg = artifact.messages[1]
            assert "Test Insight" in tool_call_msg.content
            assert "Formatted results" in tool_call_msg.content

    async def test_read_insight_not_found(self):
        """Test that not found insight raises MaxToolRetryableError."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=False)
        context_manager.artifacts = MagicMock()
        context_manager.artifacts.aget_insight_with_source = AsyncMock(return_value=None)

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        with pytest.raises(MaxToolRetryableError) as exc_info:
            await tool._arun_impl({"kind": "insight", "insight_id": "nonexistent", "execute": False})

        assert "nonexistent" in str(exc_info.value)

    async def test_read_insight_default_execute_is_false(self):
        """Test that execute defaults to False when not specified."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=False)

        mock_query = AssistantTrendsQuery(series=[AssistantTrendsEventsNode(name="$pageview")])
        mock_content = VisualizationArtifactContent(
            name="Test Insight",
            description=None,
            query=mock_query,
        )

        context_manager.artifacts = MagicMock()
        context_manager.artifacts.aget_insight_with_source = AsyncMock(
            return_value=(mock_content, ArtifactSource.INSIGHT)
        )

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        # Don't pass execute, it should default to False and return schema only
        result, artifact = await tool._arun_impl({"kind": "insight", "insight_id": "abc123"})

        assert artifact is None
        assert "Test Insight" in result
        assert "Query definition" in result

    async def test_read_insight_uses_fallback_name_when_none(self):
        """Test that insight name falls back to 'Insight {id}' when name is None."""
        team = MagicMock()
        user = MagicMock()
        state = AssistantState(messages=[], root_tool_call_id=str(uuid4()))
        context_manager = MagicMock()
        context_manager.check_user_has_billing_access = AsyncMock(return_value=False)

        mock_query = AssistantTrendsQuery(series=[AssistantTrendsEventsNode(name="$pageview")])
        mock_content = VisualizationArtifactContent(
            name=None,
            description=None,
            query=mock_query,
        )

        context_manager.artifacts = MagicMock()
        context_manager.artifacts.aget_insight_with_source = AsyncMock(
            return_value=(mock_content, ArtifactSource.INSIGHT)
        )

        tool = await ReadDataTool.create_tool_class(
            team=team,
            user=user,
            state=state,
            context_manager=context_manager,
        )

        result, artifact = await tool._arun_impl({"kind": "insight", "insight_id": "abc123", "execute": False})

        assert "Insight abc123" in result
