# -*- coding: utf-8 -*-
"""
根据近期对话和已有个人记忆，回填 community.member_profiles 中的
性格特点 / 背景信息 / 喜好偏好。

默认 dry-run，只输出结果，不写入数据库。

用法示例：
  python scripts/backfill_member_profiles_from_memory.py
  python scripts/backfill_member_profiles_from_memory.py --user-id 1234567890
  python scripts/backfill_member_profiles_from_memory.py --limit 20
  python scripts/backfill_member_profiles_from_memory.py --write
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.future import select

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.database.database import AsyncSessionLocal
from src.database.models import CommunityMemberProfile
from src.chat.features.personal_memory.services.personal_memory_service import (
    personal_memory_service,
)
from src.chat.features.world_book.services.incremental_rag_service import (
    incremental_rag_service,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


async def _load_targets(user_id: str | None, limit: int | None) -> list[CommunityMemberProfile]:
    async with AsyncSessionLocal() as session:
        stmt = select(CommunityMemberProfile).where(
            CommunityMemberProfile.discord_id.isnot(None)
        )

        if user_id:
            stmt = stmt.where(CommunityMemberProfile.discord_id == str(user_id))
        elif limit and limit > 0:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def _process_profile(
    profile: CommunityMemberProfile,
    *,
    write: bool,
) -> dict:
    user_id = str(profile.discord_id or "").strip()
    dialogue_text = personal_memory_service._build_dialogue_text(profile.history or [])

    if not user_id:
        return {"status": "skipped_missing_user_id"}

    result = await personal_memory_service.maybe_autofill_member_profile_from_memory(
        user_id=int(user_id),
        profile=profile,
        dialogue_text=dialogue_text,
        existing_summary=profile.personal_summary,
        persist=write,
        reindex_in_background=False,
    )

    if write and result.get("status") == "updated":
        await incremental_rag_service.process_community_member(result["member_id"])

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="根据近期对话和个人记忆，回填社区成员名片字段。"
    )
    parser.add_argument("--user-id", type=str, default=None, help="只处理指定 Discord ID")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    parser.add_argument(
        "--write",
        action="store_true",
        help="真正写入数据库；默认只 dry-run 输出结果",
    )
    args = parser.parse_args()

    targets = await _load_targets(args.user_id, args.limit)
    if not targets:
        log.info("没有找到可处理的名片。")
        return

    log.info(
        "开始处理成员名片回填：count=%s, mode=%s",
        len(targets),
        "write" if args.write else "dry-run",
    )

    updated = 0
    previews = 0
    skipped = 0

    for index, profile in enumerate(targets, start=1):
        result = await _process_profile(profile, write=args.write)
        status = result.get("status", "unknown")
        discord_id = str(getattr(profile, "discord_id", "") or "")
        title = str(getattr(profile, "title", "") or "")

        if status == "updated":
            updated += 1
        elif status == "preview":
            previews += 1
        else:
            skipped += 1

        log.info(
            "[%s/%s] discord_id=%s title=%s status=%s",
            index,
            len(targets),
            discord_id,
            title,
            status,
        )

        if status in {"preview", "updated"}:
            print(
                json.dumps(
                    {
                        "discord_id": discord_id,
                        "title": title,
                        "status": status,
                        "fields": result.get("fields", {}),
                        "full_text": result.get("full_text", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    log.info(
        "处理完成：updated=%s, previews=%s, skipped=%s",
        updated,
        previews,
        skipped,
    )


if __name__ == "__main__":
    asyncio.run(main())
