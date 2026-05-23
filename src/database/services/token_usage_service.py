from datetime import date
from sqlalchemy import inspect, text
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import TokenUsage


class TokenUsageService:
    @staticmethod
    async def ensure_cache_token_columns(session: AsyncSession) -> None:
        def _get_missing_columns(sync_session) -> list[str]:
            inspector = inspect(sync_session.bind)
            column_names = {
                column["name"] for column in inspector.get_columns(TokenUsage.__tablename__)
            }
            missing = []
            if "prompt_cache_hit_tokens" not in column_names:
                missing.append("prompt_cache_hit_tokens")
            if "prompt_cache_miss_tokens" not in column_names:
                missing.append("prompt_cache_miss_tokens")
            return missing

        missing_columns = await session.run_sync(_get_missing_columns)
        if not missing_columns:
            return

        if "prompt_cache_hit_tokens" in missing_columns:
            await session.execute(
                text(
                    "ALTER TABLE token_usage ADD COLUMN prompt_cache_hit_tokens INTEGER NOT NULL DEFAULT 0"
                )
            )

        if "prompt_cache_miss_tokens" in missing_columns:
            await session.execute(
                text(
                    "ALTER TABLE token_usage ADD COLUMN prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0"
                )
            )

        await session.commit()

    @staticmethod
    async def get_token_usage(
        session: AsyncSession, usage_date: date
    ) -> TokenUsage | None:
        await TokenUsageService.ensure_cache_token_columns(session)
        result = await session.execute(
            select(TokenUsage).filter(TokenUsage.date == usage_date)
        )
        return result.scalars().first()

    @staticmethod
    async def create_token_usage(
        session: AsyncSession,
        usage_date: date,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        prompt_cache_hit_tokens: int = 0,
        prompt_cache_miss_tokens: int = 0,
    ) -> TokenUsage:
        await TokenUsageService.ensure_cache_token_columns(session)
        new_usage = TokenUsage(
            date=usage_date,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_cache_hit_tokens=prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=prompt_cache_miss_tokens,
            call_count=1,
        )
        session.add(new_usage)
        await session.commit()
        return new_usage

    @staticmethod
    async def update_token_usage(
        session: AsyncSession,
        usage_record: TokenUsage,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        prompt_cache_hit_tokens: int = 0,
        prompt_cache_miss_tokens: int = 0,
    ) -> TokenUsage:
        await TokenUsageService.ensure_cache_token_columns(session)
        usage_record.input_tokens += input_tokens
        usage_record.output_tokens += output_tokens
        usage_record.total_tokens += total_tokens
        usage_record.prompt_cache_hit_tokens += prompt_cache_hit_tokens
        usage_record.prompt_cache_miss_tokens += prompt_cache_miss_tokens
        usage_record.call_count += 1
        await session.commit()
        return usage_record


token_usage_service = TokenUsageService()
