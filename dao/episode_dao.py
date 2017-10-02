#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from util.dbutil import DBUtil
from models.episode import Episode


def add_episode(episode):
    session = DBUtil.open_session()
    try:
        session.merge(episode)
        session.commit()
        return True
    except Exception as e:
        print e
        session.rollback()
        return False
    finally:
        DBUtil.close_session(session)


def add_episodes(episode_list):
    session = DBUtil.open_session()
    try:
        for episode in episode_list:
            session.merge(episode)
        session.commit()
        return True
    except Exception as e:
        print e
        session.rollback()
        return False
    finally:
        DBUtil.close_session(session)


def find_all_episodes():
    session = DBUtil.open_session()
    try:
        result = session.query(Episode).all()
        return result
    except Exception as e:
        print e
        session.rollback()
        return False
    finally:
        DBUtil.close_session(session)


def find_episodes_by_ids(episode_id_list):
    session = DBUtil.open_session()
    try:
        result = session.query(Episode).filter(Episode.episode_id.in_(episode_id_list)).all()
        return result
    except Exception as e:
        print e
        session.rollback()
        return False
    finally:
        DBUtil.close_session(session)


def find_episodes_by_bangumi(bangumi_season_id):
    session = DBUtil.open_session()
    try:
        result = session.query(Episode).filter(Episode.season_id == bangumi_season_id).order_by("index").all()
        return result
    except Exception as e:
        print e
        session.rollback()
        return False
    finally:
        DBUtil.close_session(session)
