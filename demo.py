"""
CLI demo of the query layer.

Answers the killer question from the design doc:
    "What was the second track of Minimoni's second album?"

Plus several other test queries to validate correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from query import (
    answer_track_position,
    connect,
    find_album_for_artist,
    find_artist_page,
    get_song_info,
    get_tracklist,
    resolve_title,
    resolve_title_with_aliases,
)


def fmt_album(album) -> str:
    parts = [album.page.title]
    if album.album_number is not None:
        parts.append(f"#{album.album_number}")
    if album.release_date:
        parts.append(album.release_date)
    if album.track_count is not None:
        parts.append(f"{album.track_count} tracks")
    return "  • " + " | ".join(parts)


def main() -> int:
    db = Path(__file__).parent / "helloproject.db"
    if not db.exists():
        print(f"No database at {db}. Run build_index.py first.")
        return 1

    with connect(db) as conn:
        # =========================================================
        # KILLER TEST
        # =========================================================
        print("=" * 70)
        print("KILLER TEST: 'What was the 2nd track of Minimoni's 2nd album?'")
        print("=" * 70)
        result = answer_track_position(conn, "Minimoni", album_position=2, track_position=2)
        if not result:
            print("No result returned.")
        else:
            artist = result["artist"]
            album = result.get("album")
            track = result.get("track")
            song_info = result.get("song_info")
            print(f"\nArtist: {artist.title} (id={artist.id})")
            if album:
                print(f"Album:  {album.page.title}")
                print(f"        album_number={album.album_number}  "
                      f"release={album.release_date}  "
                      f"tracks={album.track_count}")
            if track:
                print(f"\nTrack #{track.position}: {track.raw!r}")
                print(f"  section:        {track.section}")
                print(f"  linked_title:   {track.linked_title}")
                print(f"  is_karaoke:     {track.is_karaoke}")
                if song_info:
                    print(f"\n  Song page: {song_info['page'].title}")
                    if song_info.get("intro"):
                        print(f"  Intro: {song_info['intro'][:300]}...")

        # =========================================================
        # SANITY CHECK 1: List all Minimoni albums
        # =========================================================
        print("\n" + "=" * 70)
        print("SANITY: Minimoni's discography")
        print("=" * 70)
        artist = find_artist_page(conn, "Minimoni")
        if artist:
            print(f"\nArtist page: {artist.title} (id={artist.id})")
            albums = find_album_for_artist(conn, artist)
            print(f"\nAlbums ({len(albums)}):")
            for a in albums:
                print(fmt_album(a))

        # =========================================================
        # SANITY CHECK 2: Full tracklist of Minimoni Songs 2
        # =========================================================
        print("\n" + "=" * 70)
        print("SANITY: Full tracklist of 'Minimoni Songs 2'")
        print("=" * 70)
        page = resolve_title(conn, "Minimoni Songs 2")
        if page:
            tracks = get_tracklist(conn, page.id)
            print(f"\n{len(tracks)} tracks:")
            for t in tracks:
                k = " (karaoke)" if t.is_karaoke else ""
                print(f"  {t.position:2}. {t.raw[:80]}{k}")

        # =========================================================
        # SANITY CHECK 3: Title resolution with redirects
        # =========================================================
        print("\n" + "=" * 70)
        print("SANITY: Title resolution (with redirect-following)")
        print("=" * 70)
        for q in ["Minimoni", "Minimoni.", "C-ute", "℃-ute", "CRAZY ABOUT YOU",
                   "Crazy About You", "Minihamus no Ai no Uta"]:
            page, chain = resolve_title_with_aliases(conn, q)
            if page:
                chain_str = " -> ".join([q] + chain + [page.title])
                print(f"  {q!r:35s} → {page.title}  (chain: {chain_str})")
            else:
                print(f"  {q!r:35s} → NOT FOUND")

        # =========================================================
        # SANITY CHECK 4: A different album lookup — Morning Musume
        # =========================================================
        print("\n" + "=" * 70)
        print("SANITY: Morning Musume albums")
        print("=" * 70)
        artist = find_artist_page(conn, "Morning Musume")
        if artist:
            print(f"\nArtist page: {artist.title} (id={artist.id})")
            albums = find_album_for_artist(conn, artist)
            print(f"\nFirst 10 albums ({len(albums)} total):")
            for a in albums[:10]:
                print(fmt_album(a))

        # =========================================================
        # SANITY CHECK 5: Song info for CRAZY ABOUT YOU
        # =========================================================
        print("\n" + "=" * 70)
        print("SANITY: Song info for 'CRAZY ABOUT YOU'")
        print("=" * 70)
        info = get_song_info(conn, "CRAZY ABOUT YOU")
        if info:
            print(f"\nPage: {info['page'].title}")
            print(f"Alias chain: {info['alias_chain']}")
            if info.get("intro"):
                print(f"\nIntro:\n  {info['intro']}")
            print(f"\nInfobox keys: {list(info['infobox'].keys())}")
            # Show a few notable fields
            for k in ("name", "Japanese", "artist", "released", "type", "format"):
                if k in info["infobox"]:
                    print(f"  {k}: {info['infobox'][k][:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())